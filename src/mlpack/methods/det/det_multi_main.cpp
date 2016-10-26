/**
 * @file det_multi_main.cpp
 * @author Ivan Georgiev (ivan@jonan.info)
 *
 * A tool for running multi DETs on a set
 */
#include <mlpack/core.hpp>
#include <vector>
#include "dtree.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

PROGRAM_INFO("Density Estimation With Density Estimation Forest",
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by --training_file or -t) using cross-validation (with "
    "number of folds specified by --folds).  This trained density estimation "
    "tree may then be saved to a model file with the --output_model_file (-M) "
    "option."
    "\n\n"
    "The variable importances of each dimension may be saved with the "
    "--vi_file (-i) option, and the density estimates on each training point "
    "may be saved to the file specified with the --training_set_estimates_file "
    "(-e) option."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the --test_file (-T) file.  The density estimation tree used "
    "for this task will be the tree that was trained on the given training "
    "points, or a tree stored in the file given with the --input_model_file "
    "(-m) parameter.  The density estimates for the test points may be saved "
    "into the file specified with the --test_set_estimates_file (-E) option.");

// Input data file.
PARAM_STRING_IN("test_file", "A set of test points to estimate the density of.",
                "t", "");

// Input or output model.
PARAM_VECTOR_IN_REQ(string, "model_file", "File(s) containing already trained"
                "density estimation tree(s).", "m");

// Output data files.
PARAM_STRING_OUT("estimates_file", "The file in which to output the density"
    "estimates on the test set.", "e");

typedef DTree<arma::mat, int> DET;

struct noop { void operator()(...) const {} };

void Dealloc(const DET* tree) { delete tree; }

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Validate input parameters.
  if (!CLI::HasParam("model_file"))
    Log::Fatal << "You need an already built model(s), in order to "
               << "run the estimation!" << endl;
  
  vector<string> modelFiles = CLI::GetParam<vector<string>>("model_file");
  vector<DET*> models;
  
  Log::Info << modelFiles.size() << " models provided." << endl;

  size_t dataSize = 0;
  Timer::Start("models_loading");
  for (size_t i = 0;i < modelFiles.size(); ++i)
  {
    Log::Info << "Model loading " << modelFiles[i] << "...";

    DET* tree;
    data::Load(modelFiles[i], "det_model", tree, true);
    if (!tree)
      Log::Warn << " failed loading " << modelFiles[i] << endl;
    else
    {
      Log::Info << " done." << endl;
      models.push_back(tree);
      dataSize = std::max<size_t>(dataSize, tree->MaxVals().n_elem);
    }
  }
  Timer::Stop("models_loading");

  Log::Assert(dataSize > 0);
  // Now prepare the inputs and outputs.
  shared_ptr<istream> input;
  shared_ptr<ostream> output;
  
  if (!CLI::HasParam("test_file"))
  {
    Log::Info << "The estimation will operate on the standard input." << endl;
    input.reset(&cin, noop());
  }
  else
  {
    string fName = CLI::GetParam<string>("test_file");
    Log::Info << "Processing " << fName << "...";
    input.reset(new ifstream(fName, std::ifstream::in));
    if (!input->good())
      Log::Fatal << " failed!" << endl;
    else
      Log::Info << " done." << endl;
  }

  if (!CLI::HasParam("estimates_file"))
  {
    Log::Info << "The estimation will be printed on the standard output." << endl;
    output.reset(&cout, noop());
  }
  else
  {
    string fName = CLI::GetParam<string>("estimates_file");
    output.reset(new ofstream(fName, std::ofstream::out));
    if (!output->good())
      Log::Fatal << "Failed to open " << fName << " for writing!" << endl;
  }

  Timer::Start("processing");
  // Compute the density at the provided test points and output the density.

  std::string line_string;
  std::string token;
  
  while(input->good())
  {
    std::getline(*input, line_string);
    if(line_string.size() == 0)
      break;
    
    std::stringstream line_stream(line_string);
    
    arma::vec data;
    data.zeros(dataSize);

    for (size_t i = 0;!line_stream.eof() && i < dataSize; ++i)
      line_stream >> data.at(i);
    
    long double density = .0;
    size_t count = 0;
    
#ifdef _WIN32
  #pragma omp parallel for default(shared)
  for (intmax_t m = 0; m < (intmax_t)models.size(); ++m)
#else
  #pragma omp parallel for default(shared)
  for (size_t m = 0; m < models.size(); ++m)
#endif
    {
      DET* model = models[m];
      const double val = model->ComputeValue(data);
      const size_t cnt = model->End() - model->Start();
      density += (long double)val * cnt;
      count += cnt;
    }
    
    density /= count;
    
    *output << (double)density << endl;
  }
  
  Timer::Stop("processing");
  for_each(models.begin(), models.end(), Dealloc);
}

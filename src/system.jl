# System module.
module System

using RCall
using Conda

import PyCall: pyimport, pycall

export LIB_SKL_AVAILABLE,
       LIB_CRT_AVAILABLE

function check_py_dep(package::AbstractString)
  is_available = true
  try
    pyimport(package)
  catch
    try
      Conda.add(package)
      is_available = true
    catch
      is_available = false
    end
  end
  return is_available
end

function check_r_dep()
  is_available = true
  try
    R"library(caret)"
  catch
    try
      R"install.packages('caret',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('earth',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('mda',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('e1071',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('gam',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('randomForest',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('nnet',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('kernlab',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('grid',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('MASS',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('pls',repos='https://cloud.r-project.org',type='binary')"
      R"install.packages('xgboost',repos='https://cloud.r-project.org',type='binary')"
      is_available = true
    catch
      is_available = false
    end
  end
  return is_available
end

# disable support for caret/scikitlearn for faster building/development
# use julia native libraries only
LIB_SKL_AVAILABLE = false
LIB_CRT_AVAILABLE = false

### Check system for python dependencies.
#if "LOAD_SK_CARET" in keys(ENV) && ENV["LOAD_SK_CARET"] == "true"
#  LIB_SKL_AVAILABLE = check_py_dep("scikit-learn")
#  LIB_CRT_AVAILABLE = check_r_dep()
#elseif "LOAD_SK_CARET" in keys(ENV) && ENV["LOAD_SK_CARET"] == "false"
#  LIB_SKL_AVAILABLE = false
#  LIB_CRT_AVAILABLE = false
#else
#  LIB_SKL_AVAILABLE = check_py_dep("scikit-learn")
#  LIB_CRT_AVAILABLE = check_r_dep()
#end

end # module

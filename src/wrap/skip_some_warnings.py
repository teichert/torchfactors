import warnings

# silence TensorBoard internal warnings
warnings.filterwarnings(action="ignore",
                        category=DeprecationWarning,
                        module='(mlflow)|(pkg_resources)')

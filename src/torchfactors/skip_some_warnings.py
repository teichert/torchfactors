import warnings

# silence TensorBoard internal warnings
warnings.filterwarnings(action="ignore",
                        category=DeprecationWarning,
                        module='(tensorboard)|(pytorch_lightning)|(mlflow)|(pkg_resources)')

warnings.filterwarnings(action="ignore",
                        category=UserWarning,
                        module='pytorch_lightning')

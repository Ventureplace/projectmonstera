# datamancer/__init__.py

try:
    from .cleaner import smart_clean, smart_clean_extended
except ImportError as e:
    print(f"Warning: Error importing from cleaner module. {str(e)}")
    smart_clean = smart_clean_extended = None

try:
    from .splitters import smart_split
except ImportError as e:
    print(f"Warning: Error importing from splitters module. {str(e)}")
    smart_split = None

try:
    from .type_infer import infer_types, TypeInformation
except ImportError as e:
    print(f"Warning: Error importing from type_infer module. {str(e)}")
    infer_types = TypeInformation = None

try:
    from .validator import DataSchema, validate_data, validate_dataframe, generate_schema_from_dataframe
except ImportError as e:
    print(f"Warning: Error importing from validator module. {str(e)}")
    DataSchema = validate_data = validate_dataframe = generate_schema_from_dataframe = None

try:
    from .insights import generate_data_report, correlation_matrix, plot_correlation_heatmap, feature_importance
except ImportError as e:
    print(f"Warning: Error importing from insights module. {str(e)}")
    generate_data_report = correlation_matrix = plot_correlation_heatmap = feature_importance = None

try:
    from .feature_engineer import auto_features, create_interaction_features, create_polynomial_features, create_date_features
except ImportError as e:
    print(f"Warning: Error importing from feature_engineer module. {str(e)}")
    auto_features = create_interaction_features = create_polynomial_features = create_date_features = None

__all__ = []

if smart_clean and smart_clean_extended:
    __all__ += ["smart_clean", "smart_clean_extended"]

if smart_split:
    __all__ += ["smart_split"]

if infer_types and TypeInformation:
    __all__ += ["infer_types", "TypeInformation"]

if DataSchema and validate_data and validate_dataframe and generate_schema_from_dataframe:
    __all__ += ["DataSchema", "validate_data", "validate_dataframe", "generate_schema_from_dataframe"]

if generate_data_report and correlation_matrix and plot_correlation_heatmap and feature_importance:
    __all__ += ["generate_data_report", "correlation_matrix", "plot_correlation_heatmap", "feature_importance"]

if auto_features:
    __all__ += ["auto_features", "create_interaction_features", "create_polynomial_features", "create_date_features"]
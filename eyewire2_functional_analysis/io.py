"""
Written by Federico D'Agostino: https://github.com/fededagos
"""


import numpy as np
import pandas as pd


def serialize_numpy_arrays(df, verbose=False):
    """Serialize multi-dimensional numpy arrays in dataframe columns for storage in parquet."""
    df_serialized = df.copy()

    for col in df_serialized.columns:
        if df_serialized[col].dtype == "object":
            # Check if this column contains numpy arrays
            sample_non_null = df_serialized[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if isinstance(first_val, np.ndarray):
                    if verbose:
                        print(f"Serializing numpy arrays in column: {col}")
                    # Convert numpy arrays to nested lists - needed to save N-dimensional arrays to parquet
                    df_serialized[col] = df_serialized[col].apply(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )

    return df_serialized


def restore_numpy_arrays(df, verbose=False):
    """Automatically detect and restore numpy arrays from nested structures."""
    df_restored = df.copy()

    def is_nested_array_structure(val):
        """Check if a value is a nested array structure that should be converted to a numpy array.

        Args:
            val: The value to inspect.

        Returns:
            bool: ``True`` if ``val`` is a numpy object array containing arrays,
            or a nested list (list of lists/arrays).
        """

        # Check for numpy object arrays containing other arrays (from parquet loading)
        if isinstance(val, np.ndarray) and val.dtype == object:
            # Check if it contains arrays
            if val.size > 0:
                flat_val = val.flatten()
                for item in flat_val:
                    if isinstance(item, (np.ndarray, list)):
                        return True

        # Check for nested lists
        if isinstance(val, list) and len(val) > 0:
            # Check if it's a nested list (list of lists)
            if isinstance(val[0], (list, np.ndarray)):
                return True

        return False

    def convert_to_numpy_array(val):
        """Convert a nested list or numpy object array to a regular numpy array.

        Attempts ``np.stack`` first for object arrays, falling back to
        ``np.array`` on the list representation. Returns the value unchanged
        if conversion fails or if ``val`` is ``None``/NaN.

        Args:
            val: A nested list, numpy object array, or scalar value.

        Returns:
            numpy.ndarray or original value: The converted array, or ``val``
            if conversion is not possible.
        """

        # Handle numpy object arrays (from parquet)
        if isinstance(val, np.ndarray) and val.dtype == object:
            # Try to stack the arrays inside
            try:
                return np.stack(val)
            except:
                # If stacking fails, convert to list first then to array
                try:
                    nested_list = [item.tolist() if isinstance(item, np.ndarray) else item for item in val]
                    return np.array(nested_list)
                except:
                    return val

        # Handle nested lists
        if isinstance(val, list):
            try:
                return np.array(val)
            except:
                return val

        if val is None or pd.isna(val):
            return val

        return val

    for col in df_restored.columns:
        if df_restored[col].dtype == "object":
            # Check if this column contains nested array structures
            sample_non_null = df_restored[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if is_nested_array_structure(first_val):
                    if verbose:
                        print(f"Restoring numpy arrays in column: {col}")
                    df_restored[col] = df_restored[col].apply(convert_to_numpy_array)

    return df_restored
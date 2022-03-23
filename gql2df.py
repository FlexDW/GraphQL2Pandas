import textwrap
import requests
import pandas as pd
import numpy as np
import warnings
from json.decoder import JSONDecodeError
from typing import Any

NULL = np.nan

class Client:

    def __init__(
        self, url: str, headers: dict = {}, 
        field_separator: str = '.', **kwargs: Any
    ):

        self.url = url
        self.headers = headers
        self.field_separator = field_separator
        self.options = kwargs

    def query(
        self, query: str, variables: dict = None,
        operation_name: str = None, headers: dict = {},
        flatten: bool = True, schema: dict = None, 
        **kwargs: Any,
    ):

        if schema and self.validate_schema(schema) and not flatten:
            raise ValueError("Argument `flatten` must be `True` for schema to be applied.")

        body = {'query': query, 'variables': variables, 'operation_name': operation_name}
        body = {k:v for k, v in body.items() if v}

        with requests.Session() as s:
            r = s.post(
                self.url,
                json=body,
                headers={**self.headers, **headers},
                **{**self.options, **kwargs},
            )

        try:
            data = r.json()['data']
            assert r.status_code == 200
        except (KeyError, AssertionError) as e:
            raise FailedRequest(self.url, query, r.status_code, r.json(), str(e))
        except (TypeError, JSONDecodeError) as e:
            raise FailedRequest(self.url, query, r.status_code, None, str(e))            
        except Exception as e:
            raise FailedRequest(self.url, query, None, None, str(e))

        if flatten:
            df = self.flatten(data)
            if schema:
                df = self.apply_schema(df, schema)
            return df
        else:
            return data
        
    def is_empty(self, data) -> bool:
        if isinstance(data, dict):
            return all( self.is_empty(data[k]) for k in data.keys() )
        else:
            if data:
                return False
            else:
                return True


    def flatten(self, data: dict) -> pd.DataFrame:

        if self.is_empty(data):
            return pd.DataFrame()
        else:
            df = pd.json_normalize(data, sep=self.field_separator)

        i = 0
        while i < len(df.columns):

            column = df.columns[i]

            null_vals = df[column].isnull()
            list_vals = df[column].apply(lambda x: isinstance(x, list))

            if any(list_vals):
                if all(list_vals | null_vals):
                    df = self.explode_safely(df, column)
                else:
                    warnings.warn(textwrap.fill(
                        f"Flatten operation failed on column: {column}, due to unexpected values"
                        f"(i.e. list types encountered with other non-null types)."
                    ))
                    i += 1
            else:
                i += 1

        return df

    def explode_safely(self, df: pd.DataFrame, column: str) -> pd.DataFrame:

        empty = ~df[column].astype(bool)

        do_explode = df.loc[~empty]
        dont_explode = df.loc[empty]

        other_columns = [ c for c in df.columns if c != column ]
        exploded = pd.concat(
            [
                do_explode.explode(column)[other_columns].reset_index(drop=True),
                pd.json_normalize(do_explode[column].explode(), sep=self.field_separator)
            ],
            axis=1
        )

        if dont_explode.empty:
            return exploded
        else:
            return pd.concat([
                exploded,
                dont_explode[other_columns]
            ]).reset_index(drop=True)

    def validate_schema(self, schema: dict) -> bool:

        if schema is None:
            return True
        
        if not isinstance(schema, dict):
            raise ValueError(textwrap.fill(
                r"Invalid Schema: schema must be a dict of column names and corresponding types"
                r" i.e. {'column1': type1, 'column2': type2, ...}. The type specified can be any"
                r" type object or string interpretable as a numpy.dtype (e.g. float, 'float' or"
                r" 'float64')"
            ))

        for k, v in schema.items():
            try:
                pd.Series(dtype=v)
            except TypeError:
                raise ValueError(textwrap.fill(
                    f"Invalid Schema: {v} for column: {k} is not interpretable as a valid dtype."
                    r" A valid schema must be a dict of column names and corresponding types"
                    r" i.e. {'column1': type1, 'column2': type2, ...}. The type specified can be any"
                    r" type object or string interpretable as a numpy.dtype (e.g. float, 'float' or"
                    r" 'float64')"
                ))
        return True                

    def apply_schema(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:

        if df.empty:
            return pd.DataFrame(dict([ 
                (k, pd.Series(dtype=v)) for k, v in schema.items() 
            ]))

        for k, v in schema.items():
            if k not in df.columns:
                df[k] = NULL

            try:
                df[k] = df[k].astype(v)
            except TypeError:
                warnings.warn(textwrap.fill(f"column: {k} contains incompatible types for casting to dtype {v}, ignoring."))
                continue

        df = df[schema.keys()]
    
        return df

class FailedRequest(Exception):
    def __init__(self, host, query, response_status, response_content, exception):
        self.host = host
        self.query = query
        self.response_status = response_status
        self.original_exception = exception
        self.response_content = response_content

    def __str__(self):
        return (
            f"Attempted query on resource: {self.host}" 
            f"\n Query: {self.query}"
            f"\n Response status code: {self.response_status}"
            f"\n Response content: {self.response_content}"
            f"\n Original exception: {self.original_exception}"
        )

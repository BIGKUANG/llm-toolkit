#!/usr/bin/env python3
"""
================================================================================
PyTorch Profiler Parser - Process Chrome Trace JSON to CSV
================================================================================

Usage:
    python process.py --input <input.json> --output <output_prefix>

Example:
    python process.py --input result.json --output result
    
    This will generate two CSV files:
    - result_by_name.csv          : Deduplicated by operator name only
    - result_by_name_shape.csv    : Deduplicated by operator name + input shape

Description:
    This script processes PyTorch profiler JSON trace files (Chrome trace format)
    and generates CSV reports with operator statistics including:
    - Operator name, phase, timestamp, duration, thread/process IDs
    - Input shapes (extracted from args.Input Dims)
    - Input strides (extracted from args.Input Strides)
    - Input dtypes (extracted from args.Input type)
    - Call count and total duration for each unique operator
    - has_shape column indicating whether shape/stride/dtype info was extracted

JSON Generation Example:
    import torch
    x = torch.randn((1, 1), requires_grad=True)
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        y = x ** 2
        y.backward()
    print(prof)
    prof.export_chrome_trace("result.json")

================================================================================
"""

import json
import csv
import argparse
import os
from tqdm import tqdm


def format_list_without_comma(val):
    """Format a list or value without internal commas.
    
    Converts list representations to use space as internal separator instead of comma.
    This ensures that when the final output is comma-separated, each input position
    can be reliably parsed by splitting on commas.
    
    Examples:
        [1, 2, 3] -> "[1 2 3]"
        [[1, 2], [3, 4]] -> "[[1 2] [3 4]]"
        "[1, 2, 3]" (string) -> "[1 2 3]"
        "0" -> "0"
        "" -> ""
    """
    import re
    
    if val is None:
        return ''
    
    if isinstance(val, list):
        # Recursively format nested lists
        formatted_items = []
        for item in val:
            if isinstance(item, list):
                formatted_items.append(format_list_without_comma(item))
            else:
                formatted_items.append(str(item))
        return '[' + ' '.join(formatted_items) + ']'
    
    # For string values, replace commas inside brackets with spaces
    val_str = str(val)
    if '[' in val_str and ']' in val_str:
        # Replace comma+optional space with space
        return re.sub(r',\s*', ' ', val_str)
    
    return val_str


def extract_input_info(event):
    """Extract input shape, stride, and dtype from event args.
    
    This function extracts ALL values from 'Concrete Inputs' in order, including:
    - Tensor shapes (from Input Dims when Concrete Input is empty string)
    - ScalarList values (e.g., "[5 8 128]" - spaces instead of commas)
    - Scalar values (e.g., "0", "15", "True", "False")
    
    The three output lists (input_shape, input_stride, input_dtype) are aligned:
    they are output as JSON arrays with the same length.
    Internal lists use space as separator to avoid conflicts.
    
    Returns:
        dict: Contains 'input_shape', 'input_stride', 'input_dtype', 'has_shape'
    """
    result = {
        'input_shape': '',
        'input_stride': '',
        'input_dtype': '',
        'has_shape': 'N'
    }
    
    if 'args' not in event:
        return result
    
    args = event['args']
    
    # Check if Concrete Inputs exists - if so, has_shape is Y
    if 'Concrete Inputs' not in args:
        return result
    
    concrete_inputs = args['Concrete Inputs']
    if not isinstance(concrete_inputs, list):
        return result
    
    # Mark has_shape as Y since Concrete Inputs exists
    result['has_shape'] = 'Y'
    
    input_dims = args.get('Input Dims', [])
    input_strides = args.get('Input Strides', [])
    input_types = args.get('Input type', [])
    
    # Ensure all lists have the same length as concrete_inputs
    num_inputs = len(concrete_inputs)
    
    # Pad lists to match length
    while len(input_dims) < num_inputs:
        input_dims.append([])
    while len(input_strides) < num_inputs:
        input_strides.append([])
    while len(input_types) < num_inputs:
        input_types.append('')
    
    shape_parts = []
    stride_parts = []
    dtype_parts = []
    
    for i in range(num_inputs):
        concrete_val = concrete_inputs[i]
        dim_val = input_dims[i] if i < len(input_dims) else []
        stride_val = input_strides[i] if i < len(input_strides) else []
        type_val = input_types[i] if i < len(input_types) else ''
        
        # Determine shape value:
        # - If concrete_val is empty string "", it's a tensor placeholder -> use Input Dims
        # - Otherwise use the concrete value directly (ScalarList, Scalar, etc.)
        if concrete_val == '' and dim_val:
            # Tensor: use Input Dims as shape (formatted without commas)
            shape_parts.append(format_list_without_comma(dim_val))
        else:
            # Use concrete value directly, formatting lists without commas
            shape_parts.append(format_list_without_comma(concrete_val))
        
        # Determine stride value:
        # - If stride_val is non-empty list, format it without commas
        # - Otherwise use empty string
        if stride_val:
            stride_parts.append(format_list_without_comma(stride_val))
        else:
            stride_parts.append('')
        
        # Determine dtype value:
        # - Use Input type directly (no commas expected in type names)
        dtype_parts.append(str(type_val) if type_val else '')
    
    # Output as JSON arrays for better readability and parsing
    result['input_shape'] = json.dumps(shape_parts, ensure_ascii=False)
    result['input_stride'] = json.dumps(stride_parts, ensure_ascii=False)
    result['input_dtype'] = json.dumps(dtype_parts, ensure_ascii=False)
    
    return result


def process_events(events):
    """Process events and return two dictionaries: by name only and by name+shape."""
    
    # Dict for deduplication by name only
    by_name = {}
    # Dict for deduplication by name + shape
    by_name_shape = {}
    
    for event in tqdm(events, desc="Processing events"):
        # Skip events without required fields
        if 'name' not in event or 'dur' not in event:
            continue
        
        name = event['name']
        pid = event.get('pid', '')
        tid = event.get('tid', '')
        
        # Extract shape, stride, dtype info
        input_info = extract_input_info(event)
        input_shape = input_info['input_shape']
        input_stride = input_info['input_stride']
        input_dtype = input_info['input_dtype']
        has_shape = input_info['has_shape']
        
        # Key for name-only deduplication
        key_name = f"{name}|{pid}|{tid}"
        # Key for name+shape deduplication
        key_name_shape = f"{name}|{pid}|{tid}|{input_shape}"
        
        # Process by name only
        if key_name not in by_name:
            by_name[key_name] = {
                'name': name,
                'cat': event.get('cat', ''),
                'ph': event.get('ph', ''),
                'ts': event.get('ts', 0),
                'dur': event['dur'],
                'tid': tid,
                'pid': pid,
                'input_shape': input_shape,
                'input_stride': input_stride,
                'input_dtype': input_dtype,
                'has_shape': has_shape,
                'call_num': 1
            }
        else:
            by_name[key_name]['call_num'] += 1
            by_name[key_name]['dur'] += event['dur']
            # Update info if current one has shape and previous didn't
            if has_shape == 'Y' and by_name[key_name]['has_shape'] == 'N':
                by_name[key_name]['input_shape'] = input_shape
                by_name[key_name]['input_stride'] = input_stride
                by_name[key_name]['input_dtype'] = input_dtype
                by_name[key_name]['has_shape'] = 'Y'
        
        # Process by name + shape
        if key_name_shape not in by_name_shape:
            by_name_shape[key_name_shape] = {
                'name': name,
                'cat': event.get('cat', ''),
                'ph': event.get('ph', ''),
                'ts': event.get('ts', 0),
                'dur': event['dur'],
                'tid': tid,
                'pid': pid,
                'input_shape': input_shape,
                'input_stride': input_stride,
                'input_dtype': input_dtype,
                'has_shape': has_shape,
                'call_num': 1
            }
        else:
            by_name_shape[key_name_shape]['call_num'] += 1
            by_name_shape[key_name_shape]['dur'] += event['dur']
    
    return by_name, by_name_shape


def write_csv(data_dict, output_path, keys):
    """Write data to CSV file."""
    values = list(data_dict.values())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(values)
    
    print(f"Written {len(values)} rows to {output_path}")


def main(args):
    print(f"Loading JSON file: {args.input}")
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict):
        if 'traceEvents' in data:
            events = data['traceEvents']
        else:
            try:
                events = [data[k] for k in sorted(data.keys(), key=int)]
            except (ValueError, KeyError):
                events = list(data.values())
    else:
        events = data
    
    print(f"Total events to process: {len(events)}")
    
    # Process events
    by_name, by_name_shape = process_events(events)
    
    # Output columns
    keys = ['name', 'cat', 'ph', 'ts', 'dur', 'tid', 'pid', 'input_shape', 'input_stride', 'input_dtype', 'has_shape', 'call_num']
    
    # Generate output paths
    output_base = args.output
    if output_base.endswith('.csv'):
        output_base = output_base[:-4]
    if output_base.endswith('.xlsx'):
        output_base = output_base[:-5]
    
    output_by_name = f"{output_base}_by_name.csv"
    output_by_name_shape = f"{output_base}_by_name_shape.csv"
    
    # Write CSV files
    print("\nGenerating CSV files...")
    write_csv(by_name, output_by_name, keys)
    write_csv(by_name_shape, output_by_name_shape, keys)
    
    print("\nSummary:")
    print(f"  - Unique operators (by name): {len(by_name)}")
    print(f"  - Unique operators (by name+shape): {len(by_name_shape)}")
    print(f"\nOutput files:")
    print(f"  - {output_by_name}")
    print(f"  - {output_by_name_shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process PyTorch profiler JSON trace to CSV reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process.py --input result.json --output result
    python process.py --input trace.json --output ./output/trace
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file path (Chrome trace format)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file prefix (will generate _by_name.csv and _by_name_shape.csv)')
    args = parser.parse_args()
    main(args)

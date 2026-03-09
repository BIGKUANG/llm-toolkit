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


def extract_input_info(event):
    """Extract input shape, stride, and dtype from event args.
    
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
    has_any_info = False
    
    # Extract Input Dims (shape)
    if 'Input Dims' in args:
        input_dims = args['Input Dims']
        non_empty_dims = [str(d) for d in input_dims if d]
        if non_empty_dims:
            result['input_shape'] = '|'.join(non_empty_dims)
            has_any_info = True
    
    # Extract Input Strides
    if 'Input Strides' in args:
        input_strides = args['Input Strides']
        non_empty_strides = [str(s) for s in input_strides if s]
        if non_empty_strides:
            result['input_stride'] = '|'.join(non_empty_strides)
            has_any_info = True
    
    # Extract Input type (dtype)
    if 'Input type' in args:
        input_types = args['Input type']
        non_empty_types = [str(t) for t in input_types if t and t != '']
        if non_empty_types:
            result['input_dtype'] = '|'.join(non_empty_types)
            has_any_info = True
    
    if has_any_info:
        result['has_shape'] = 'Y'
    
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

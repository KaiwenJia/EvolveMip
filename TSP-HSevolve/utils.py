import  subprocess
import re
import json
"""
The load data
"""
def load_jsonl(filepath):
    """Loads a JSONL (JSON Lines) file and returns a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}")
                    print(f"Error details: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
    print(f"Successfully loaded {len(data)} items from {filepath}")
    return data

#+tsp_load
def load_tsp(filepath):
    """
    Parses a .tsp file and returns a list of dictionaries.
    The function extracts problem metadata and the node coordinates,
    then formats it into a single dictionary within a list.
    """
    data = {}
    node_coords_section = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Join all lines into a single string for easier regex or split operations
        content = "".join(lines)
        
        # Extract the problem description as a single string
        problem_description = content.strip()
        
        # Prepare the final data structure
        output_data = [{
            'en_question': problem_description
        }]

        print(f"Successfully loaded 1 item from {filepath}")
        return output_data
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []
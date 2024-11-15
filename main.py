# main.py
from data_preprocessing import preprocess_data
from data_analysis import analyze_data
from graph_generation import generate_graphs
from inference_making import make_inferences_with_graphs
from critique_agent import critique_outputs

def main(file_path: str):
    preprocessed_data = preprocess_data(file_path)
    analysis_results = analyze_data(preprocessed_data.data, preprocessed_data.context)
    graphs = generate_graphs(preprocessed_data.data, preprocessed_data.context)
    inferences = make_inferences_with_graphs(preprocessed_data.data, graphs.graph_paths)
    critique = critique_outputs(graphs.graph_paths, inferences.text)
    
    # Output results
    print("Data Analysis:")
    print(analysis_results.description)
    print("\nInferences:")
    print(inferences.text)
    print("\nCritique:")
    print(critique.feedback)

if __name__ == "__main__":
    data_file = 'data/test_file.xlsx'  # Update with your data file path
    main(data_file)
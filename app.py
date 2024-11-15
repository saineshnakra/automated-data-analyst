# app.py
import streamlit as st
from data_preprocessing import preprocess_data
from data_analysis import analyze_data
from graph_generation import generate_graphs
from inference_making import make_inferences_with_graphs

def main():
    st.set_page_config(page_title="Data Analyst", layout="wide")
    st.title("Data Analyst")
    
    # Create two columns for file upload and context
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload your CSV file")
        uploaded_file = st.file_uploader("", type=['csv'])
    
    with col2:
        st.markdown("### Data Description")
        data_description = st.text_area("", 
            placeholder="Describe your dataset (e.g., 'Sales data containing order information, quantities, and prices')",
            value="")

    if uploaded_file and data_description:
        preprocessed_data = preprocess_data(uploaded_file, data_description)
        if preprocessed_data:
            with st.spinner('Processing...'):
                analysis_results = analyze_data(preprocessed_data.data, preprocessed_data.context)
                graphs = generate_graphs(preprocessed_data.data, preprocessed_data.context)
                inferences = make_inferences_with_graphs(
                    preprocessed_data.data, 
                    graphs.graph_paths,
                    preprocessed_data.context  # Pass the context here
                )

            # Display inferences at the top
            st.header("Key Insights")
            st.info(inferences.text)

            # Visualizations
            st.markdown("### Visualizations")
            for graph_path in graphs.graph_paths:
                st.image(graph_path, use_container_width=True)

            # Detailed Analysis
            st.markdown("### Detailed Analysis")
            st.text(analysis_results.description)

        else:
            st.error("Preprocessing failed.")
    else:
        st.info("Please upload a file and provide a data description to begin analysis.")

if __name__ == "__main__":
    main()
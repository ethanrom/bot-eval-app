import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def main():
    st.set_page_config(page_title="ROUGE & BLEU Scores App", page_icon=":pencil:")
    st.title("ROUGE & BLEU Scores App")
    st.write("This app calculates ROUGE and BLEU scores and visualizes the results from a CSV file containing ground truth and bot prediction answers for a set of questions.")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV file into Pandas DataFrame
        df = pd.read_csv(uploaded_file)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []

        for index, row in df.iterrows():
            ground_truth = row.iloc[1].strip()
            prediction = row.iloc[2].strip()

            # Calculate ROUGE scores
            rouge_scores = scorer.score(ground_truth, prediction)
            rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
            rouge2_scores.append(rouge_scores["rouge2"].fmeasure)
            rougeL_scores.append(rouge_scores["rougeL"].fmeasure)

            # Calculate BLEU score
            bleu_score = sentence_bleu([ground_truth.split()], prediction.split())
            bleu_scores.append(bleu_score)

        df["ROUGE-1"] = rouge1_scores
        df["ROUGE-2"] = rouge2_scores
        df["ROUGE-L"] = rougeL_scores
        df["BLEU"] = bleu_scores

        mean_rouge1 = df["ROUGE-1"].mean()
        mean_rouge2 = df["ROUGE-2"].mean()
        mean_rougeL = df["ROUGE-L"].mean()
        mean_bleu = df["BLEU"].mean()

        st.write("## Mean Scores")
        st.write(f"- ROUGE-1 Score: {mean_rouge1:.4f}")
        st.write(f"- ROUGE-2 Score: {mean_rouge2:.4f}")
        st.write(f"- ROUGE-L Score: {mean_rougeL:.4f}")
        st.write(f"- BLEU Score: {mean_bleu:.4f}")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
        ax1.boxplot([rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores], labels=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU'])
        ax1.set_title('Scores Distribution')
        ax1.set_ylabel('Score')

        ax2.bar(['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU'], [mean_rouge1, mean_rouge2, mean_rougeL, mean_bleu])
        ax2.set_title('Mean Scores')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0,1])

        st.write("## Scores Distribution")
        st.pyplot(fig)

if __name__ == "__main__":
    main()

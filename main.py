import streamlit as st
import pandas as pd
from tv_schedule_ga import read_csv_to_dict, genetic_algorithm

st.title("📺 TV Scheduling using Genetic Algorithm")

# --- Load Dataset ---
ratings_file = "program_ratings.csv"
ratings = read_csv_to_dict(ratings_file)

# --- Sidebar Inputs ---
st.sidebar.header("Genetic Algorithm Parameters")

co_r = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
mut_r = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)
gen = st.sidebar.number_input("Generations", min_value=10, max_value=500, value=100, step=10)
pop = st.sidebar.number_input("Population Size", min_value=10, max_value=100, value=50, step=10)

if st.button("Run Genetic Algorithm"):
    schedule_df, total_rating = genetic_algorithm(ratings, crossover_rate=co_r, mutation_rate=mut_r, generations=gen, pop_size=pop)

    st.subheader("🗓️ Generated Schedule")
    st.dataframe(schedule_df)
    st.success(f"✅ Total Ratings: {total_rating:.2f}")

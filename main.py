import streamlit as st
import pandas as pd
import csv
import random
from io import StringIO

# ============================
# 1. Read CSV and prepare data
# ============================
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# ============================
# 2. GA Functions (imported from your original file)
# ============================
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, ratings, generations, population_size, crossover_rate, mutation_rate, elitism_size, all_programs):
    def fitness_function_local(schedule):
        return fitness_function(schedule, ratings)

    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function_local(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# ============================
# 3. Streamlit Interface
# ============================
st.title("📺 TV Scheduling using Genetic Algorithm")

uploaded_file = st.file_uploader("Upload the program_ratings.csv file", type="csv")

if uploaded_file is not None:
    # Read CSV from upload
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    reader = csv.reader(stringio)
    header = next(reader)
    program_ratings = {}
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings

    all_programs = list(program_ratings.keys())
    all_time_slots = list(range(6, 24))

    st.subheader("⚙️ Genetic Algorithm Parameters")
    generations = st.number_input("Number of Generations", 10, 500, 100)
    population_size = st.number_input("Population Size", 10, 100, 50)
    crossover_rate = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
    mutation_rate = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)
    elitism_size = st.number_input("Elitism Size", 1, 5, 2)

    if st.button("Run Genetic Algorithm"):
        # Initialize schedule
        initial_schedule = all_programs.copy()
        random.shuffle(initial_schedule)

        # Run GA
        best_schedule = genetic_algorithm(initial_schedule, program_ratings, generations, population_size,
                                          crossover_rate, mutation_rate, elitism_size, all_programs)

        # Compute fitness
        total_rating = fitness_function(best_schedule, program_ratings)

        # Display results
        st.subheader("🗓️ Final Optimal Schedule")
        time_slots_to_show = all_time_slots[:len(best_schedule)]
        result_df = pd.DataFrame({
        "Time Slot": [f"{t:02d}:00" for t in time_slots_to_show],
        "Program": best_schedule + ["(Empty)"] * (len(all_time_slots) - len(best_schedule))
        })


        st.dataframe(result_df)
        st.success(f"⭐ Total Ratings: {total_rating:.2f}")

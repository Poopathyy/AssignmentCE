import streamlit as st
import pandas as pd
import csv
import random

# ================================
# 1. Read CSV into dictionary
# ================================
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# ================================
# 2. Core Genetic Algorithm Functions
# ================================
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules, ratings):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule, ratings)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

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
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule, ratings), reverse=True)
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

# ================================
# 3. Streamlit Interface
# ================================
st.title("📺 TV Scheduling using Genetic Algorithm")

uploaded_file = st.file_uploader("Upload your program_ratings.csv file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded CSV
    ratings = {}
    reader = csv.reader(uploaded_file.getvalue().decode("utf-8").splitlines())
    header = next(reader)
    for row in reader:
        program = row[0]
        ratings[program] = [float(x) for x in row[1:]]

    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # 6:00–23:00

    st.sidebar.header("⚙️ GA Parameters")
    generations = st.sidebar.number_input("Generations", 10, 500, 100)
    population_size = st.sidebar.number_input("Population Size", 10, 200, 50)
    crossover_rate = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
    mutation_rate = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)
    elitism_size = st.sidebar.number_input("Elitism Size", 1, 5, 2)

    if st.button("Run Genetic Algorithm"):
        # Initialize schedule
        initial_schedule = all_programs.copy()
        random.shuffle(initial_schedule)

        # Brute force best schedule (for your original code logic)
        all_possible_schedules = initialize_pop(all_programs, all_time_slots)
        initial_best_schedule = finding_best_schedule(all_possible_schedules, ratings)

        # GA process
        rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
        genetic_schedule = genetic_algorithm(initial_best_schedule, ratings, generations, population_size, crossover_rate, mutation_rate, elitism_size, all_programs)
        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

        # Make sure final schedule matches time slots
        num_slots = len(all_time_slots)
        if len(final_schedule) < num_slots:
            final_schedule += ["(Empty)"] * (num_slots - len(final_schedule))
        elif len(final_schedule) > num_slots:
            final_schedule = final_schedule[:num_slots]

        total_rating = fitness_function(final_schedule, ratings)

        # Display results
        st.subheader("🗓️ Final Optimal Schedule")
        result_df = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": final_schedule
        })

        st.dataframe(result_df)
        st.success(f"⭐ Total Ratings: {total_rating:.2f}")

else:
    st.info("👆 Please upload the program_ratings.csv file to begin.")

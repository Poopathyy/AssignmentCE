import streamlit as st
import pandas as pd
import csv
import random
from io import StringIO

# ============================
# 1. Helper Functions
# ============================
def read_csv_to_dict(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    reader = csv.reader(stringio)
    header = next(reader)
    program_ratings = {}
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings
    return program_ratings

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
# 2. Streamlit Interface
# ============================
st.title("📺 TV Scheduling using Genetic Algorithm")
st.subheader("Run and Compare 3 Separate Trials")

uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv file", type="csv")

if uploaded_file:
    program_ratings = read_csv_to_dict(uploaded_file)
    all_programs = list(program_ratings.keys())
    all_time_slots = list(range(6, 24))

    # Loop through 3 trials
    for trial in range(1, 4):
        st.markdown(f"### 🧪 Trial {trial}")

        generations = st.number_input(f"Trial {trial} - Generations", 10, 500, 100, key=f"gen{trial}")
        population_size = st.number_input(f"Trial {trial} - Population Size", 10, 100, 50, key=f"pop{trial}")
        crossover_rate = st.slider(f"Trial {trial} - Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01, key=f"co{trial}")
        mutation_rate = st.slider(f"Trial {trial} - Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01, key=f"mu{trial}")
        elitism_size = st.number_input(f"Trial {trial} - Elitism Size", 1, 5, 2, key=f"el{trial}")

        if st.button(f"Run Trial {trial}"):
            initial_schedule = all_programs.copy()
            random.shuffle(initial_schedule)

            best_schedule = genetic_algorithm(initial_schedule, program_ratings, generations,
                                              population_size, crossover_rate, mutation_rate,
                                              elitism_size, all_programs)

            total_rating = fitness_function(best_schedule, program_ratings)

            # Display results in a table
            result_df = pd.DataFrame({
                "Time Slot": [f"{t:02d}:00" for t in all_time_slots[:len(best_schedule)]],
                "Program": best_schedule
            })

            st.dataframe(result_df)
            st.success(f"⭐ Trial {trial} - Total Ratings: {total_rating:.2f}")

            st.markdown("---")

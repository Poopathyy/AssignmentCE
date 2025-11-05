import streamlit as st
import pandas as pd
import csv
import random

# ================================
# 1. Read CSV (no upload)
# ================================
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

# Load dataset
file_path = "program_ratings.csv"  # must be in same folder
ratings = read_csv_to_dict(file_path)

# ================================
# 2. Genetic Algorithm Functions
# ================================
def fitness_function(schedule):
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

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
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

def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size, all_programs):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
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
st.title("📺 TV Scheduling using Genetic Algorithm (Three Trials)")

# Fixed GA parameters
GEN = 100
POP = 50
EL_S = 2
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))  # 06:00–23:00

# Display fixed parameters
st.sidebar.header("⚙️ Fixed GA Parameters")
st.sidebar.write(f"Generations (GEN): {GEN}")
st.sidebar.write(f"Population Size (POP): {POP}")
st.sidebar.write(f"Elitism Size (EL_S): {EL_S}")

# Input 3 different trials for CO_R and MUT_R
st.sidebar.header("🧪 Adjustable Parameters for Each Trial")
trial_params = []
for i in range(1, 4):
    st.sidebar.subheader(f"Trial {i}")
    co_r = st.sidebar.slider(f"Trial {i} - Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01, key=f"co_{i}")
    mut_r = st.sidebar.slider(f"Trial {i} - Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01, key=f"mut_{i}")
    trial_params.append((co_r, mut_r))

if st.button("Run All Three Trials"):
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)

    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

    for i, (co_r, mut_r) in enumerate(trial_params, start=1):
        st.subheader(f"🧩 Trial {i}: CO_R = {co_r}, MUT_R = {mut_r}")

        genetic_schedule = genetic_algorithm(
            initial_best_schedule,
            generations=GEN,
            population_size=POP,
            crossover_rate=co_r,
            mutation_rate=mut_r,
            elitism_size=EL_S,
            all_programs=all_programs
        )

        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

        # Ensure full time coverage (6–23)
        num_slots = len(all_time_slots)
        if len(final_schedule) < num_slots:
            final_schedule += ["(Empty)"] * (num_slots - len(final_schedule))
        elif len(final_schedule) > num_slots:
            final_schedule = final_schedule[:num_slots]

        total_rating = fitness_function(final_schedule)

        # Display each trial's result
        result_df = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": final_schedule
        })

        st.dataframe(result_df, use_container_width=True)
        st.success(f"⭐ Total Ratings (Trial {i}): {total_rating:.2f}")
        st.markdown("---")

else:
    st.info("👈 Set crossover and mutation rates for each trial, then click **Run All Three Trials**.")

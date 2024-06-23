import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

plt.style.use("ggplot")


def read_data():
    with open("./results/results_no_gauss.json") as f:
        data = json.load(f)

    return data


def theoretical_complexity(method, matrix_size, iterations):
    if method == "ConjugateGradient":
        return iterations * matrix_size**2
    elif method == "Gradient":
        return iterations * matrix_size**2
    elif method == "Jacobi":
        return iterations * matrix_size**2
    else:
        return iterations * matrix_size**2


def aggiungi_linea_complessita(ax, x, y, label, color):
    ax.plot(x, y, linestyle="--", label=label, color=color)


matrix_sizes = {"spa1": 1000, "vem1": 1681, "spa2": 3000, "vem2": 2601}

colors = list(sns.color_palette("Set1", n_colors=8))

data = read_data()

index = []

for key in data.keys():
    for key_2 in data[key].keys():
        for key_3 in data[key][key_2].keys():
            index.append((key, key_2, key_3))

df = pd.DataFrame(
    columns=[
        "std_memory",
        "errors",
        "iterations",
        "std_time",
        "std_errors",
        "memory",
        "times",
        "std_iterations",
    ],
    index=pd.MultiIndex.from_tuples(index),
)

for key in data.keys():
    for key_2 in data[key].keys():
        for key_3 in data[key][key_2].keys():
            df.loc[(key, key_2, key_3)] = data[key][key_2][key_3]

df.reset_index(inplace=True)
df.columns = [
    "matrice",
    "tolleranza",
    "metodo",
    "std_memory",
    "errors",
    "iterations",
    "std_time",
    "std_errors",
    "memory",
    "times",
    "std_iterations",
]

df = df.astype(
    {
        "std_memory": "float64",
        "errors": "float64",
        "iterations": "int64",
        "std_time": "float64",
        "std_errors": "float64",
        "memory": "float64",
        "times": "float64",
        "std_iterations": "float64",
    }
)

matrici = df["matrice"].unique()
fig, axes = plt.subplots(nrows=1, ncols=len(matrici))

# Itera attraverso ogni matrice
for idx, matrice in enumerate(matrici):
    # Filtra i dati per la matrice corrente
    df_matrice = df[df["matrice"] == matrice]

    for col, metodo in enumerate(df_matrice["metodo"].unique()):
        df_metodo = df_matrice[df_matrice["metodo"] == metodo]
        df_metodo = df_metodo.sort_values(by="tolleranza")
        sns.lineplot(
            ax=axes[idx],
            data=df_metodo,
            x="tolleranza",
            y="times",
            label=metodo,
            color=colors[col],
        )
        axes[idx].fill_between(
            df_metodo["tolleranza"],
            df_metodo["times"] - df_metodo["std_time"],
            df_metodo["times"] + df_metodo["std_time"],
            alpha=0.3,
            color=colors[col],
        )
        aggiungi_linea_complessita(
            ax=axes[idx],
            x=df_metodo["tolleranza"],
            y=theoretical_complexity(
                metodo, matrix_sizes[matrice], df_metodo["iterations"]
            ),
            label=f"Complessità {metodo}",
            color=colors[col],
        )

    axes[idx].set_title(f"Tempo medio - {matrice}")
    axes[idx].set_xlabel("Tolleranza")
    axes[idx].set_ylabel("Tempo medio")
    axes[idx].set_yscale("log")
    plt.gca().invert_xaxis()

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=len(matrici))

# Itera attraverso ogni matrice
for idx, matrice in enumerate(matrici):
    # Filtra i dati per la matrice corrente
    df_matrice = df[df["matrice"] == matrice]

    for col, metodo in enumerate(df_matrice["metodo"].unique()):
        df_metodo = df_matrice[df_matrice["metodo"] == metodo]
        df_metodo = df_metodo.sort_values(by="tolleranza")
        sns.lineplot(
            ax=axes[idx],
            data=df_metodo,
            x="tolleranza",
            y="memory",
            label=metodo,
            color=colors[col],
        )
        axes[idx].fill_between(
            df_metodo["tolleranza"],
            df_metodo["memory"] - df_metodo["std_memory"],
            df_metodo["memory"] + df_metodo["std_memory"],
            alpha=0.3,
            color=colors[col],
        )

    axes[idx].set_title(f"Memoria utilizzata - {matrice}")
    axes[idx].set_xlabel("Tolleranza")
    axes[idx].set_ylabel("Memoria utilizzata")
    axes[idx].set_yscale("log")

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=len(matrici))

# Itera attraverso ogni matrice
for idx, matrice in enumerate(matrici):
    # Filtra i dati per la matrice corrente
    df_matrice = df[df["matrice"] == matrice]

    for col, metodo in enumerate(df_matrice["metodo"].unique()):
        df_metodo = df_matrice[df_matrice["metodo"] == metodo]
        df_metodo = df_metodo.sort_values(by="tolleranza")
        sns.lineplot(
            ax=axes[idx],
            data=df_metodo,
            x="tolleranza",
            y="errors",
            label=metodo,
            color=colors[col],
        )
        axes[idx].fill_between(
            df_metodo["tolleranza"],
            df_metodo["errors"] - df_metodo["std_errors"],
            df_metodo["errors"] + df_metodo["std_errors"],
            alpha=0.3,
            color=colors[col],
        )

    axes[idx].set_title(f"Errore - {matrice}")
    axes[idx].set_xlabel("Tolleranza")
    axes[idx].set_ylabel("Errore")


plt.show()

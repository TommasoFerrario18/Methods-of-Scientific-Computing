import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


# Funzione per leggere i file CSV e combinarli in un unico DataFrame
def load_data():
    new_df = pd.DataFrame(
        columns=[
            "metodo",
            "matrice",
            "tolleranza",
            "tempo_risoluzione",
            "memoria_utilizzata",
            "errore",
        ]
    )

    metodi = ["Jacobi", "GaussSeidel", "Gradient", "ConjugateGradient"]
    matrici = ["spa1", "spa2", "vem1", "vem2"]
    tol = [1e-5, 1e-7, 1e-9, 1e-11]

    for matrice in matrici:
        times = pd.read_csv(f"./results/times_{matrice}.csv")
        memory = pd.read_csv(f"./results/memory_{matrice}.csv")
        error = pd.read_csv(f"./results/errors_{matrice}.csv")
        for m in metodi:
            for t in tol:
                new_df = pd.concat(
                    [
                        new_df,
                        pd.DataFrame(
                            {
                                "metodo": m,
                                "matrice": matrice,
                                "tolleranza": t,
                                "tempo_risoluzione": times[m][tol.index(t)],
                                "memoria_utilizzata": memory[m][tol.index(t)],
                                "errore": error[m][tol.index(t)],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    return new_df


data = load_data()

# Visualizza le prime righe del DataFrame
print(data.head())

# Grafico del tempo di risoluzione rispetto alla tolleranza per ogni metodo e matrice
g = sns.FacetGrid(data, col="matrice", hue="metodo", col_wrap=4, height=4, sharey=False)
g.map(sns.lineplot, "tolleranza", "tempo_risoluzione", marker="o")
g.add_legend()
g.set_titles(col_template="Matrice: {col_name}")
g.set(xscale="log", yscale="log")
g.set_axis_labels("Tolleranza", "Tempo di Risoluzione (s)")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Tempo di Risoluzione vs Tolleranza")
plt.gca().invert_xaxis()
plt.show()

# Grafico della memoria utilizzata rispetto alla tolleranza per ogni metodo e matrice
g = sns.FacetGrid(data, col="matrice", hue="metodo", col_wrap=4, height=4, sharey=False)
g.map(sns.lineplot, "tolleranza", "memoria_utilizzata", marker="o")
g.add_legend()
g.set_titles(col_template="Matrice: {col_name}")
g.set(xscale="log")
g.set_axis_labels("Tolleranza", "Memoria Utilizzata (MB)")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Memoria Utilizzata vs Tolleranza")
plt.gca().invert_xaxis()
plt.show()

# Grafico dell'errore rispetto alla tolleranza per ogni metodo e matrice
g = sns.FacetGrid(data, col="matrice", hue="metodo", col_wrap=4, height=4, sharey=False)
g.map(sns.lineplot, "tolleranza", "errore", marker="o")
g.add_legend()
g.set_titles(col_template="Matrice: {col_name}")
g.set(xscale="log", yscale="log")
g.set_axis_labels("Tolleranza", "Errore")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Errore vs Tolleranza")
plt.gca().invert_xaxis()
plt.show()

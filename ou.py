import numpy as np
import matplotlib.pyplot as plt


def genera_tabella_grafica(risultati):
    """Genera una figura Matplotlib contenente la tabella riassuntiva formattata."""
    fig, ax = plt.subplots(figsize=(10, 1.8 + len(risultati) * 0.6))
    ax.axis("off")

    col_labels = [
        "Dataset",
        "Condition Number",
        "Rango SVD",
        "RMSE Willems [m]",
        "RMSE SVD [m]",
    ]
    cell_text = []

    for r in risultati:
        cell_text.append(
            [
                r["etichetta"],
                f"{r['condition_number']:.2e}",
                str(r["rango_svd"]),
                f"{r['rmse_willems']:.4f}",
                f"{r['rmse_svd']:.4f}",
            ]
        )

    # Creazione della tabella grafica
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 2.0)  # Dimensione e spaziatura delle celle

    # Styling professionale (Intestazione scura, righe alternate)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f77b4")  # Blu primario per l'header
            cell.set_text_props(color="white", weight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f8f9fa")  # Grigio leggero a righe alternate
            else:
                cell.set_facecolor("white")

    plt.title(
        "TABELLA RIASSUNTIVA MODEL OU (10+10)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.tight_layout()


def analizza_ou(nome_file, etichetta_rumore, rango_svd):

    data = np.load(nome_file)
    X = data["states"]
    Y = data["observations"]

    lengths = data.get("lengths", np.full(X.shape[0], X.shape[1]))

    # PARAMETRI FINESTRA (Configurazione 10+10)
    N_passato = 10
    N_futuro = 10
    dim_misura = 2
    L = N_passato + N_futuro  # L = 20

    # 1. COSTRUZIONE MATRICI TRAINING (800 traiettorie)
    H_Y_list = []
    H_Z_list = []

    for i in range(800):
        Ti = lengths[i]
        traiettoria_y = Y[i, :Ti, :]

        for k in range(Ti - L + 1):
            passato = traiettoria_y[k : k + N_passato, :]
            futuro = traiettoria_y[k + N_passato : k + N_passato + N_futuro, :]
            H_Y_list.append(passato.flatten())
            H_Z_list.append(futuro.flatten())

    H_Y = np.array(H_Y_list).T
    H_Z = np.array(H_Z_list).T

    # 2. SVD E SPETTRO SINGOLARE
    U, S, Vt = np.linalg.svd(H_Y, full_matrices=False)
    condition_number = S[0] / S[-1]

    # Grafico Spettro Singolare
    plt.figure(figsize=(8, 4))
    plt.semilogy(S, "b.-", linewidth=1.5, label="Valori singolari")
    plt.axvline(x=rango_svd, color="r", linestyle="--", label=f"Rango troncamento = {rango_svd}")
    plt.title(f"Spettro Singolare (OU) - {etichetta_rumore}\nCondition Number: {condition_number:.2e}")
    plt.xlabel("Indice (max 20)")
    plt.ylabel("Valore singolare (scala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()

    # 3. PSEUDOINVERSE (Willems con pinv vs SVD Troncata)
    H_Y_pinv_willems = np.linalg.pinv(H_Y)

    U_r = U[:, :rango_svd]
    S_r = S[:rango_svd]
    Vt_r = Vt[:rango_svd, :]
    S_r_inv = np.diag(1.0 / S_r)
    H_Y_pinv_svd = Vt_r.T @ S_r_inv @ U_r.T

    # 4. TEST SU TRAIETTORIE 800-999
    istanti_iniziali = [40, 60, 80, 100, 120]

    mse_willems_list = []
    mse_svd_list = []

    # Dati per i grafici di esempio (850, t=60 e 850, t=80)
    target_grafici = [(850, 60), (850, 80), (850, 100)]
    dati_grafici = {}

    for i_test in range(800, 1000):
        Ti_test = lengths[i_test]
        traiettoria_test_y = Y[i_test, :Ti_test, :]
        traiettoria_test_x = X[i_test, :Ti_test, :]

        for ist_ini in istanti_iniziali:
            if ist_ini + L > Ti_test:
                continue

            passato_osservato = traiettoria_test_y[ist_ini : ist_ini + N_passato, :]
            nuovo_Y_vec = passato_osservato.flatten()

            futuro_reale_y = traiettoria_test_y[
                ist_ini + N_passato : ist_ini + N_passato + N_futuro, :
            ]
            futuro_reale_x = traiettoria_test_x[
                ist_ini + N_passato : ist_ini + N_passato + N_futuro, 0:2
            ]

            # Predizione Willems Standard
            g_willems = H_Y_pinv_willems @ nuovo_Y_vec
            pred_willems = (H_Z @ g_willems).reshape(N_futuro, dim_misura)
            mse_willems_list.append(np.mean((futuro_reale_x - pred_willems) ** 2))

            # Predizione SVD Troncata
            g_svd = H_Y_pinv_svd @ nuovo_Y_vec
            pred_svd = (H_Z @ g_svd).reshape(N_futuro, dim_misura)
            mse_svd_list.append(np.mean((futuro_reale_x - pred_svd) ** 2))

            # Salviamo le traiettorie selezionate per i grafici
            if (i_test, ist_ini) in target_grafici:
                dati_grafici[(i_test, ist_ini)] = {
                    "passato": passato_osservato,
                    "futuro_x": futuro_reale_x,
                    "futuro_y": futuro_reale_y,
                    "pred_willems": pred_willems,
                    "pred_svd": pred_svd,
                }

    # CALCOLO RMSE GLOBALE
    rmse_willems = np.sqrt(np.mean(mse_willems_list))
    rmse_svd = np.sqrt(np.mean(mse_svd_list))

    # 5. GRAFICI CONFRONTO VISIVO
    for (traj_idx, t_idx), data_grafico in dati_grafici.items():
        plt.figure(figsize=(9, 5))
        plt.plot(data_grafico["passato"][:, 0], data_grafico["passato"][:, 1], "bo-", alpha=0.5, label="Passato Y")
        plt.plot(data_grafico["futuro_y"][:, 0], data_grafico["futuro_y"][:, 1], "go-", alpha=0.3, label="Futuro Y (rumoroso)")
        plt.plot(data_grafico["futuro_x"][:, 0], data_grafico["futuro_x"][:, 1], "k-", linewidth=2.5, label="Futuro X (vero)")
        plt.plot(data_grafico["pred_willems"][:, 0], data_grafico["pred_willems"][:, 1], "mX--", linewidth=1.5, label="Willems Std")
        plt.plot(data_grafico["pred_svd"][:, 0], data_grafico["pred_svd"][:, 1], "rD-.", linewidth=2, label=f"SVD (rango {rango_svd})")
        plt.title(f"Predizione OU - {etichetta_rumore}\n(Traiettoria {traj_idx}, t={t_idx})")
        plt.xlabel("Posizione x [m]")
        plt.ylabel("Posizione y [m]")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()

    return {
        'etichetta': etichetta_rumore,
        'condition_number': condition_number,
        'rango_svd': rango_svd,
        'rmse_willems': rmse_willems,
        'rmse_svd': rmse_svd
    }


# ==============================================================================
# ELENCO DATASET

database_list = [
    ("data/trajectory_dataset_ou_vx_8_vy_0_fixed_std_0m_dt_1.npz", "std = 0 m (Noiseless)", 6),
    ("data/trajectory_dataset_ou_vx_8_vy_0_fixed_std_5m_dt_1.npz", "std = 5 m", 3),
    ("data/trajectory_dataset_ou_vx_8_vy_0_fixed_std_10m_dt_1.npz", "std = 10 m", 3),
]

risultati = []

for percorso, etichetta, rango in database_list:
    try:
        res = analizza_ou(percorso, etichetta, rango)
        risultati.append(res)
    except FileNotFoundError:
        print(f"File non trovato: {percorso}")

# TABELLA RIASSUNTIVA (TESTUALE + GRAFICA)
if risultati:
    # 1. Stampa su terminale
    print("\n" + "=" * 75)
    print("TABELLA RIASSUNTIVA MODEL OU (10+10)")
    print("=" * 75)
    print(f"{'Dataset':<22} {'Cond. Num.':<15} {'Rango':<8} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 75)
    for r in risultati:
        print(f"{r['etichetta']:<22} {r['condition_number']:<15.2e} {r['rango_svd']:<8} {r['rmse_willems']:<15.4f} {r['rmse_svd']:<15.4f}")
    print("-" * 75)

    # 2. Generazione Figura Grafica della Tabella
    genera_tabella_grafica(risultati)

plt.show()
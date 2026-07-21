import numpy as np
import matplotlib.pyplot as plt


def analizza_ct_neg(nome_file, etichetta_rumore, rango_svd):

    data = np.load(nome_file)
    X = data["states"]
    Y = data["observations"]

    lengths = data.get("lengths", np.full(X.shape[0], X.shape[1]))

    # PARAMETRI FINESTRA (10+10)
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
    plt.title(f"Spettro Singolare CT (w < 0, Curva a Destra) - {etichetta_rumore}\nCondition Number: {condition_number:.2e}")
    plt.xlabel("Indice (max 20)")
    plt.ylabel("Valore singolare (scala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()

    # 3. PSEUDOINVERSE (Willems vs SVD)
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

    passato_grafico = None
    futuro_x_grafico = None
    futuro_y_grafico = None
    pred_willems_grafico = None
    pred_svd_grafico = None

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

            # Salviamo per il grafico di esempio (Traiettoria 850, t=60)
            if i_test == 850 and ist_ini == 60:
                passato_grafico = passato_osservato
                futuro_x_grafico = futuro_reale_x
                futuro_y_grafico = futuro_reale_y
                pred_willems_grafico = pred_willems
                pred_svd_grafico = pred_svd

    rmse_willems = np.sqrt(np.mean(mse_willems_list))
    rmse_svd = np.sqrt(np.mean(mse_svd_list))

    # 5. GRAFICO CONFRONTO VISIVO
    if passato_grafico is not None:
        plt.figure(figsize=(9, 5))
        plt.plot(passato_grafico[:, 0], passato_grafico[:, 1], "bo-", alpha=0.5, label="Passato Y")
        plt.plot(futuro_y_grafico[:, 0], futuro_y_grafico[:, 1], "go-", alpha=0.3, label="Futuro Y (rumoroso)")
        plt.plot(futuro_x_grafico[:, 0], futuro_x_grafico[:, 1], "k-", linewidth=2.5, label="Futuro X (vero)")
        plt.plot(pred_willems_grafico[:, 0], pred_willems_grafico[:, 1], "mX--", linewidth=1.5, label="Willems Std")
        plt.plot(pred_svd_grafico[:, 0], pred_svd_grafico[:, 1], "rD-.", linewidth=2, label=f"SVD (rango {rango_svd})")
        plt.title(f"Predizione CT (w < 0, Curva a Destra) - {etichetta_rumore}\n(Traiettoria 850, t=60)")
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



database_list = [
    ("data/trajectory_dataset_ct_turn_m3deg_fixed_std_0m_dt_1.npz", "std = 0 m (Noiseless)", 6),
    ("data/trajectory_dataset_ct_turn_m3deg_fixed_std_5m_dt_1.npz", "std = 5 m", 4),
    ("data/trajectory_dataset_ct_turn_m3deg_fixed_std_10m_dt_1.npz", "std = 10 m", 4),
]

risultati = []

for percorso, etichetta, rango in database_list:
    try:
        res = analizza_ct_neg(percorso, etichetta, rango)
        risultati.append(res)
    except FileNotFoundError:
        print(f"File non trovato: {percorso}")

# TABELLA RIASSUNTIVA FINALE
if risultati:
    print("\n" + "=" * 75)
    print("TABELLA RIASSUNTIVA MODEL CT (w < 0 | CURVA A DESTRA)")
    print("=" * 75)
    print(f"{'Dataset':<22} {'Cond. Num.':<15} {'Rango':<8} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 75)
    for r in risultati:
        print(f"{r['etichetta']:<22} {r['condition_number']:<15.2e} {r['rango_svd']:<8} {r['rmse_willems']:<15.4f} {r['rmse_svd']:<15.4f}")
    print("-" * 75)

plt.show()
import numpy as np
import matplotlib.pyplot as plt


def crea_tabella_grafica_classe(etichetta_rumore, condition_number, rango_svd, errori_dettaglio, rmse_glob_w, rmse_glob_s):

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    ax.axis('tight')

    col_labels = ['Classe', 'N Campioni', 'RMSE Willems [m]', 'RMSE SVD [m]']
    table_data = []

    tutti_w, tutti_s = [], []
    for classe in ['ou', 'ct_positive', 'ct_negative']:
        n = len(errori_dettaglio[classe]['willems'])
        if n > 0:
            rw = np.sqrt(np.mean(errori_dettaglio[classe]['willems']))
            rs = np.sqrt(np.mean(errori_dettaglio[classe]['svd']))
            table_data.append([classe, str(n), f"{rw:.4f}", f"{rs:.4f}"])
            tutti_w.extend(errori_dettaglio[classe]['willems'])
            tutti_s.extend(errori_dettaglio[classe]['svd'])
        else:
            table_data.append([classe, "N/A", "N/A", "N/A"])

    table_data.append(['GLOBALE', str(len(tutti_w)), f"{rmse_glob_w:.4f}", f"{rmse_glob_s:.4f}"])

    tabla = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    tabla.scale(1, 1.8)
    tabla.set_fontsize(11)

    for (i, j), cell in tabla.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2c3e50')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif i == len(table_data):
            cell.set_facecolor('#ecf0f1')
            cell.get_text().set_weight('bold')

    plt.title(f"Risultati Dettagliati: {etichetta_rumore}\nCond. Num: {condition_number:.2e} | Rango SVD: {rango_svd}",
              fontsize=12, pad=15, weight='bold')
    plt.tight_layout()


def confronto_three_model(nome_file, etichetta_rumore, rango_svd):

    data = np.load(nome_file)
    X = data["states"]
    Y = data["observations"]
    mask = data["mask"]
    lengths = data["lengths"]
    trajectory_model_ids = data["trajectory_model_ids"]

    N_passato = 10
    N_futuro = 10
    dim_misura = 2
    L = N_passato + N_futuro

    # ---------------------------------------------------------
    # COSTRUZIONE H_Y / H_Z (training: traiettorie 0-799)
    # ---------------------------------------------------------
    H_Y_list, H_Z_list = [], []
    for i in range(800):
        Ti = lengths[i]
        maschera_i = mask[i].astype(bool)
        if np.sum(maschera_i) != Ti:
            raise ValueError(f"Incoerenza mask/lengths traiettoria {i}")

        traiettoria_y = Y[i, :Ti, :]
        for k in range(Ti - L + 1):
            passato = traiettoria_y[k: k + N_passato, :]
            futuro = traiettoria_y[k + N_passato: k + N_passato + N_futuro, :]
            H_Y_list.append(passato.flatten())
            H_Z_list.append(futuro.flatten())

    H_Y = np.array(H_Y_list).T
    H_Z = np.array(H_Z_list).T

    # ---------------------------------------------------------
    # SVD E ANALISI SPETTRALE
    # ---------------------------------------------------------
    U, S, Vt = np.linalg.svd(H_Y, full_matrices=False)
    condition_number = S[0] / S[-1]

    plt.figure(figsize=(8, 4))
    plt.semilogy(range(1, len(S) + 1), S, "b.-", linewidth=1.5, label="Valori singolari")
    plt.axvline(x=rango_svd, color="r", linestyle="--", label=f"Rango scelto = {rango_svd}")
    plt.title(f"Spettro Singolare (3 modelli) - {etichetta_rumore}\nCondition Number: {condition_number:.2e}")
    plt.xlabel("Indice")
    plt.ylabel("Valore singolare (scala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()

    # ---------------------------------------------------------
    # PSEUDOINVERSE
    # ---------------------------------------------------------
    H_Y_pinv_willems = np.linalg.pinv(H_Y)

    U_r = U[:, :rango_svd]
    S_r = S[:rango_svd]
    Vt_r = Vt[:rango_svd, :]
    H_Y_pinv_svd = Vt_r.T @ np.diag(1.0 / S_r) @ U_r.T

    # ---------------------------------------------------------
    # TEST SU TRAIETTORIE 800-999
    # ---------------------------------------------------------
    istanti_iniziali = [40, 60, 80, 100, 120]

    errori = {
        'ou': {'willems': [], 'svd': []},
        'ct_positive': {'willems': [], 'svd': []},
        'ct_negative': {'willems': [], 'svd': []},
    }

    # Struttura dati per memorizzare sia la traiettoria 870 sia la 950
    traiettorie_target = [870, 950]
    istante_target = 80
    grafici_dati = {}

    for i_test in range(800, 1000):
        Ti_test = lengths[i_test]
        maschera_test = mask[i_test].astype(bool)
        if np.sum(maschera_test) != Ti_test:
            raise ValueError(f"Incoerenza mask/lengths traiettoria test {i_test}")

        traiettoria_test_y = Y[i_test, :Ti_test, :]
        traiettoria_test_x = X[i_test, :Ti_test, :]
        tipo_traiettoria = str(trajectory_model_ids[i_test])

        if tipo_traiettoria not in errori:
            continue

        for ist_ini in istanti_iniziali:
            if ist_ini + L > Ti_test:
                continue

            passato_osservato = traiettoria_test_y[ist_ini: ist_ini + N_passato, :]
            nuovo_Y_vec = passato_osservato.flatten()

            futuro_reale_y = traiettoria_test_y[
                ist_ini + N_passato: ist_ini + N_passato + N_futuro, :
            ]
            futuro_reale_x = traiettoria_test_x[
                ist_ini + N_passato: ist_ini + N_passato + N_futuro, 0:2
            ]

            g_willems = H_Y_pinv_willems @ nuovo_Y_vec
            pred_willems = (H_Z @ g_willems).reshape(N_futuro, dim_misura)
            mse_willems = np.mean((futuro_reale_x - pred_willems) ** 2)

            g_svd = H_Y_pinv_svd @ nuovo_Y_vec
            pred_svd = (H_Z @ g_svd).reshape(N_futuro, dim_misura)
            mse_svd = np.mean((futuro_reale_x - pred_svd) ** 2)

            errori[tipo_traiettoria]['willems'].append(mse_willems)
            errori[tipo_traiettoria]['svd'].append(mse_svd)

            # Salvataggio dati per i grafici richiesti (870 e 900)
            if i_test in traiettorie_target and ist_ini == istante_target:
                grafici_dati[i_test] = {
                    'passato': passato_osservato,
                    'futuro_x': futuro_reale_x,
                    'futuro_y': futuro_reale_y,
                    'pred_willems': pred_willems,
                    'pred_svd': pred_svd,
                    'tipo': tipo_traiettoria
                }

    # ---------------------------------------------------------
    # STAMPA RISULTATI CONSOLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Dataset: {etichetta_rumore}")
    print("=" * 70)
    print(f"Condition number H_Y: {condition_number:.2e}")
    print(f"Rango SVD: {rango_svd}")
    print()
    print(f"{'Classe':<15} {'N campioni':<12} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 57)

    tutti_w, tutti_s = [], []
    for classe in ['ou', 'ct_positive', 'ct_negative']:
        n = len(errori[classe]['willems'])
        if n > 0:
            rw = np.sqrt(np.mean(errori[classe]['willems']))
            rs = np.sqrt(np.mean(errori[classe]['svd']))
            print(f"{classe:<15} {n:<12} {rw:<15.4f} {rs:<15.4f}")
            tutti_w.extend(errori[classe]['willems'])
            tutti_s.extend(errori[classe]['svd'])
        else:
            print(f"{classe:<15} {'N/A':<12} {'N/A':<15} {'N/A':<15}")

    rmse_glob_w = np.sqrt(np.mean(tutti_w)) if tutti_w else float('nan')
    rmse_glob_s = np.sqrt(np.mean(tutti_s)) if tutti_s else float('nan')

    print("-" * 57)
    print(f"{'GLOBALE':<15} {len(tutti_w):<12} {rmse_glob_w:<15.4f} {rmse_glob_s:<15.4f}")

    # ---------------------------------------------------------
    # TABELLA GRAFICA
    # ---------------------------------------------------------
    crea_tabella_grafica_classe(etichetta_rumore, condition_number, rango_svd, errori, rmse_glob_w, rmse_glob_s)

    # ---------------------------------------------------------
    # GRAFICI TRAIETTORIE (870 e 900)
    # ---------------------------------------------------------
    for id_traj in sorted(grafici_dati.keys()):
        dati = grafici_dati[id_traj]
        plt.figure(figsize=(9, 5.5))
        plt.plot(dati['passato'][:, 0], dati['passato'][:, 1], "bo-", alpha=0.5, label="Passato Y")
        plt.plot(dati['futuro_y'][:, 0], dati['futuro_y'][:, 1], "go-", alpha=0.5, label="Futuro Y (rumoroso)")
        plt.plot(dati['futuro_x'][:, 0], dati['futuro_x'][:, 1], "k-", linewidth=2.5, label="Futuro X (vero)")
        plt.plot(dati['pred_willems'][:, 0], dati['pred_willems'][:, 1], "mX--", linewidth=1.5, label="Willems Std")
        plt.plot(dati['pred_svd'][:, 0], dati['pred_svd'][:, 1], "rD-.", linewidth=2, label=f"SVD (rango {rango_svd})")
        
        plt.title(f"Confronto Willems vs SVD (3 modelli) - {etichetta_rumore}\n"
                  f"(Traiettoria {id_traj} [{dati['tipo']}], t={istante_target})")
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
        'rmse_willems': rmse_glob_w,
        'rmse_svd': rmse_glob_s,
    }


def mostra_tabella_riassuntiva_finale(risultati_globali):
    """
    Crea la finestra grafica finale con la Tabella Riassuntiva dei 3 Dataset.
    """
    print("\n")
    print("=" * 90)
    print("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS SVD (3 MODELLI)")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Cond. Num.':<15} {'Rango':<8} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 90)
    for r in risultati_globali:
        print(f"{r['etichetta']:<15} {r['condition_number']:<15.2e} {r['rango_svd']:<8} "
              f"{r['rmse_willems']:<15.4f} {r['rmse_svd']:<15.4f}")
    print("-" * 90)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    ax.axis('tight')

    col_labels = ['Dataset', 'Cond. Num.', 'Rango SVD', 'RMSE Willems [m]', 'RMSE SVD [m]']
    table_data = []

    for r in risultati_globali:
        table_data.append([
            r['etichetta'],
            f"{r['condition_number']:.2e}",
            str(r['rango_svd']),
            f"{r['rmse_willems']:.4f}",
            f"{r['rmse_svd']:.4f}"
        ])

    tabla = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    tabla.scale(1, 2.0)
    tabla.set_fontsize(11)

    for (i, j), cell in tabla.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1b4f72')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f2f4f4')

    plt.title("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS SVD (3 MODELLI)",
              fontsize=13, pad=18, weight='bold')
    plt.tight_layout()


database_list = [
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_0m_dt_1.npz", "std = 0 m (Noiseless)", 16),
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_3m_dt_1.npz", "std = 3 m", 12),
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_5m_dt_1.npz", "std = 5 m", 12),
]

risultati_globali = []
for percorso, etichetta, rango in database_list:
    try:
        risultato = confronto_three_model(percorso, etichetta, rango)
        risultati_globali.append(risultato)
    except FileNotFoundError:
        print(f"\n[ERRORE] File non trovato: {percorso}")

if risultati_globali:
    mostra_tabella_riassuntiva_finale(risultati_globali)

plt.show()
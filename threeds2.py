import numpy as np
import matplotlib.pyplot as plt


def crea_tabella_grafica_classe(etichetta_rumore, condition_number, K_best, errori_dettaglio, rmse_glob_w, rmse_glob_k):

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.axis('off')
    ax.axis('tight')

    col_labels = ['Classe', 'N Campioni', 'RMSE Willems Std [m]', f'RMSE K-Best (K={K_best}) [m]']
    table_data = []

    tutti_w, tutti_k = [], []
    for classe in ['ou', 'ct_positive', 'ct_negative']:
        n = len(errori_dettaglio[classe]['willems'])
        if n > 0:
            rw = np.sqrt(np.mean(errori_dettaglio[classe]['willems']))
            rk = np.sqrt(np.mean(errori_dettaglio[classe]['k_best']))
            table_data.append([classe, str(n), f"{rw:.4f}", f"{rk:.4f}"])
            tutti_w.extend(errori_dettaglio[classe]['willems'])
            tutti_k.extend(errori_dettaglio[classe]['k_best'])
        else:
            table_data.append([classe, "N/A", "N/A", "N/A"])

    table_data.append(['GLOBALE', str(len(tutti_w)), f"{rmse_glob_w:.4f}", f"{rmse_glob_k:.4f}"])

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

    plt.title(f"Risultati Dettagliati (K-Best): {etichetta_rumore}\nCond. Num: {condition_number:.2e} | K selezionate: {K_best}",
              fontsize=12, pad=15, weight='bold')
    plt.tight_layout()


def confronto_three_model_kbest(nome_file, etichetta_rumore, K_best=500):

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
    # COSTRUZIONE H_Y, H_Z (coordinate ASSOLUTE, come in tutta la tesi)
    # E Delta_H_Y (derivata discreta del passato, usata SOLO per la
    # ricerca dei k vicini piu' prossimi, mai per la predizione).
    # In piu': Label_H_Y, la classe di ciascuna colonna, usata SOLO
    # per la diagnostica (non entra nel calcolo della predizione).
    # Training: traiettorie 0-799
    # ---------------------------------------------------------
    H_Y_list, H_Z_list, Delta_H_Y_list, Label_list = [], [], [], []

    for i in range(800):
        Ti = lengths[i]
        maschera_i = mask[i].astype(bool)
        if np.sum(maschera_i) != Ti:
            raise ValueError(f"Incoerenza mask/lengths traiettoria {i}")

        classe_traiettoria = str(trajectory_model_ids[i])
        traiettoria_y = Y[i, :Ti, :]

        for k in range(Ti - L + 1):
            passato = traiettoria_y[k: k + N_passato, :]
            futuro = traiettoria_y[k + N_passato: k + N_passato + N_futuro, :]

            H_Y_list.append(passato.flatten())
            H_Z_list.append(futuro.flatten())

            delta_passato = np.diff(passato, axis=0)  # solo per il confronto ||ΔY - ΔY_i||
            Delta_H_Y_list.append(delta_passato.flatten())

            Label_list.append(classe_traiettoria)

    H_Y = np.array(H_Y_list).T          # (p*N_passato, N_windows) = (20, N_windows)
    H_Z = np.array(H_Z_list).T          # (p*N_futuro, N_windows)  = (20, N_windows)
    Delta_H_Y = np.array(Delta_H_Y_list).T  # (p*(N_passato-1), N_windows) = (18, N_windows)
    Label_H_Y = np.array(Label_list)    # (N_windows,) etichetta di ogni colonna, solo diagnostica

    # Condition number di H_Y (per riferimento, come nelle altre sezioni)
    S = np.linalg.svd(H_Y, compute_uv=False)
    condition_number = S[0] / S[-1]

    # Pseudoinversa Willems standard, IDENTICA a quella usata in tutta
    # la tesi (coordinate assolute, nessun centramento)
    H_Y_pinv_willems = np.linalg.pinv(H_Y)

    # ---------------------------------------------------------
    # TEST SU TRAIETTORIE 800-999
    # ---------------------------------------------------------
    istanti_iniziali = [40, 60, 80, 100, 120]

    errori = {
        'ou': {'willems': [], 'k_best': []},
        'ct_positive': {'willems': [], 'k_best': []},
        'ct_negative': {'willems': [], 'k_best': []},
    }

    # Diagnostica: per ciascuna classe REALE della query, quanti dei k
    # vicini selezionati appartengono a ciascuna classe (composizione media)
    diagnostica = {
        classe_query: {'ou': 0, 'ct_positive': 0, 'ct_negative': 0, 'n_query': 0}
        for classe_query in ['ou', 'ct_positive', 'ct_negative']
    }

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
            nuovo_Y_vec = passato_osservato.flatten()  # Y, coordinate assolute

            futuro_reale_y = traiettoria_test_y[
                ist_ini + N_passato: ist_ini + N_passato + N_futuro, :
            ]
            futuro_reale_x = traiettoria_test_x[
                ist_ini + N_passato: ist_ini + N_passato + N_futuro, 0:2
            ]

            # ---------------------------------------------------------
            # 1. PREDIZIONE WILLEMS STANDARD (rango pieno, tutte le colonne)
            # ---------------------------------------------------------
            g_willems = H_Y_pinv_willems @ nuovo_Y_vec
            pred_willems = (H_Z @ g_willems).reshape(N_futuro, dim_misura)
            mse_willems = np.mean((futuro_reale_x - pred_willems) ** 2)

            # ---------------------------------------------------------
            # 2. PREDIZIONE K-BEST (algoritmo del prof)
            #    Y -> ΔY -> ricerca dei k vicini piu' prossimi (norma su ΔY)
            #    -> H_Y^k, H_Z^k -> g^k = (H_Y^k)^+ Y -> Ẑ = H_Z^k g^k
            # ---------------------------------------------------------
            delta_query = np.diff(passato_osservato, axis=0).flatten()

            distanze = np.linalg.norm(Delta_H_Y - delta_query[:, None], axis=0)
            idx_k = np.argpartition(distanze, K_best)[:K_best]

            H_Y_k = H_Y[:, idx_k]
            H_Z_k = H_Z[:, idx_k]

            g_k = np.linalg.pinv(H_Y_k) @ nuovo_Y_vec   # Y, non ΔY: come da formula del prof
            pred_kbest = (H_Z_k @ g_k).reshape(N_futuro, dim_misura)
            mse_kbest = np.mean((futuro_reale_x - pred_kbest) ** 2)

            errori[tipo_traiettoria]['willems'].append(mse_willems)
            errori[tipo_traiettoria]['k_best'].append(mse_kbest)

            # ---------------------------------------------------------
            # DIAGNOSTICA: composizione reale dei k vicini selezionati
            # ---------------------------------------------------------
            etichette_vicini = Label_H_Y[idx_k]
            diagnostica[tipo_traiettoria]['n_query'] += 1
            for classe_vicino in ['ou', 'ct_positive', 'ct_negative']:
                diagnostica[tipo_traiettoria][classe_vicino] += np.sum(etichette_vicini == classe_vicino)

            if i_test in traiettorie_target and ist_ini == istante_target:
                grafici_dati[i_test] = {
                    'passato': passato_osservato,
                    'futuro_x': futuro_reale_x,
                    'futuro_y': futuro_reale_y,
                    'pred_willems': pred_willems,
                    'pred_kbest': pred_kbest,
                    'tipo': tipo_traiettoria
                }

    # ---------------------------------------------------------
    # STAMPA DIAGNOSTICA COMPOSIZIONE VICINI
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"[DIAGNOSTICA] Composizione media dei {K_best} vicini selezionati, per classe reale della query")
    print("-" * 70)
    print(f"{'Classe query':<15} {'% vicini ou':<14} {'% vicini ct_pos':<17} {'% vicini ct_neg':<17}")
    for classe_query in ['ou', 'ct_positive', 'ct_negative']:
        n_q = diagnostica[classe_query]['n_query']
        if n_q > 0:
            tot = n_q * K_best
            pct_ou = 100 * diagnostica[classe_query]['ou'] / tot
            pct_ctp = 100 * diagnostica[classe_query]['ct_positive'] / tot
            pct_ctn = 100 * diagnostica[classe_query]['ct_negative'] / tot
            print(f"{classe_query:<15} {pct_ou:<14.1f} {pct_ctp:<17.1f} {pct_ctn:<17.1f}")
    print("-" * 70)

    # ---------------------------------------------------------
    # STAMPA RISULTATI CONSOLE
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Dataset: {etichetta_rumore} | K-Best = {K_best}")
    print("=" * 70)
    print(f"Condition number H_Y: {condition_number:.2e}")
    print()
    print(f"{'Classe':<15} {'N campioni':<12} {'RMSE Willems':<15} {'RMSE K-Best':<15}")
    print("-" * 57)

    tutti_w, tutti_k = [], []
    for classe in ['ou', 'ct_positive', 'ct_negative']:
        n = len(errori[classe]['willems'])
        if n > 0:
            rw = np.sqrt(np.mean(errori[classe]['willems']))
            rk = np.sqrt(np.mean(errori[classe]['k_best']))
            print(f"{classe:<15} {n:<12} {rw:<15.4f} {rk:<15.4f}")
            tutti_w.extend(errori[classe]['willems'])
            tutti_k.extend(errori[classe]['k_best'])
        else:
            print(f"{classe:<15} {'N/A':<12} {'N/A':<15} {'N/A':<15}")

    rmse_glob_w = np.sqrt(np.mean(tutti_w)) if tutti_w else float('nan')
    rmse_glob_k = np.sqrt(np.mean(tutti_k)) if tutti_k else float('nan')

    print("-" * 57)
    print(f"{'GLOBALE':<15} {len(tutti_w):<12} {rmse_glob_w:<15.4f} {rmse_glob_k:<15.4f}")

    # ---------------------------------------------------------
    # TABELLA GRAFICA
    # ---------------------------------------------------------
    crea_tabella_grafica_classe(etichetta_rumore, condition_number, K_best, errori, rmse_glob_w, rmse_glob_k)

    # ---------------------------------------------------------
    # GRAFICI QUALITATIVI (traiettorie 870 e 950, t=80)
    # ---------------------------------------------------------
    for id_traj in sorted(grafici_dati.keys()):
        dati = grafici_dati[id_traj]
        plt.figure(figsize=(9, 5.5))
        plt.plot(dati['passato'][:, 0], dati['passato'][:, 1], "bo-", alpha=0.5, label="Passato Y")
        plt.plot(dati['futuro_y'][:, 0], dati['futuro_y'][:, 1], "go-", alpha=0.5, label="Futuro Y (rumoroso)")
        plt.plot(dati['futuro_x'][:, 0], dati['futuro_x'][:, 1], "k-", linewidth=2.5, label="Futuro X (vero)")
        plt.plot(dati['pred_willems'][:, 0], dati['pred_willems'][:, 1], "mX--", linewidth=1.5, label="Willems Std")
        plt.plot(dati['pred_kbest'][:, 0], dati['pred_kbest'][:, 1], "rD-.", linewidth=2, label=f"K-Best (K={K_best})")

        plt.title(f"Confronto Willems Std vs K-Best - {etichetta_rumore}\n"
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
        'K_best': K_best,
        'rmse_willems': rmse_glob_w,
        'rmse_kbest': rmse_glob_k,
    }


def mostra_tabella_riassuntiva_finale(risultati_globali):

    print("\n" + "=" * 90)
    print("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS K-BEST")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Cond. Num.':<15} {'K Best':<8} {'RMSE Willems':<15} {'RMSE K-Best':<15}")
    print("-" * 90)
    for r in risultati_globali:
        print(f"{r['etichetta']:<15} {r['condition_number']:<15.2e} {r['K_best']:<8} "
              f"{r['rmse_willems']:<15.4f} {r['rmse_kbest']:<15.4f}")
    print("-" * 90)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    ax.axis('tight')

    col_labels = ['Dataset', 'Cond. Num.', 'K-Best', 'RMSE Willems [m]', 'RMSE K-Best [m]']
    table_data = []

    for r in risultati_globali:
        table_data.append([
            r['etichetta'],
            f"{r['condition_number']:.2e}",
            str(r['K_best']),
            f"{r['rmse_willems']:.4f}",
            f"{r['rmse_kbest']:.4f}"
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

    plt.title("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS K-BEST",
              fontsize=13, pad=18, weight='bold')
    plt.tight_layout()


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
database_list = [
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_0m_dt_1.npz", "std = 0 m (Noiseless)", 500),
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_3m_dt_1.npz", "std = 3 m", 500),
    ("data/trajectory_dataset_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_5m_dt_1.npz", "std = 5 m", 500),
]

risultati_globali = []
for percorso, etichetta, k_val in database_list:
    try:
        risultato = confronto_three_model_kbest(percorso, etichetta, K_best=k_val)
        risultati_globali.append(risultato)
    except FileNotFoundError:
        print(f"\n[ERRORE] File non trovato: {percorso}")

if risultati_globali:
    mostra_tabella_riassuntiva_finale(risultati_globali)

plt.show()
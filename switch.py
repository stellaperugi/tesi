import numpy as np
import matplotlib.pyplot as plt


def classifica_finestra(modelli_finestra):
    """
    Classifica la finestra PASSATO in base ai modelli attivi al suo interno.
    Restituisce 'ou', 'ct_positive', 'ct_negative' se il passato e' interamente
    in un solo regime, altrimenti 'switch' se attraversa una transizione.
    """
    modelli_unici = set(modelli_finestra)
    modelli_unici.discard('pad')
    if modelli_unici == {'ou'}:
        return 'ou'
    elif modelli_unici == {'ct_positive'}:
        return 'ct_positive'
    elif modelli_unici == {'ct_negative'}:
        return 'ct_negative'
    else:
        return 'switch'


def confronto_switching3(nome_file, etichetta_rumore, rango_svd):

    data = np.load(nome_file)
    X = data["states"]
    Y = data["observations"]
    mask = data["mask"]
    lengths = data["lengths"]
    active_model_ids = data["active_model_ids"]  

    N_passato = 25
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
    plt.title(f"Spettro Singolare (Switching 3 modelli) - {etichetta_rumore}\nCondition Number: {condition_number:.2e}")
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
    # TEST SU TRAIETTORIE 800-999, PIU' ISTANTI INIZIALI
    # ---------------------------------------------------------
    istanti_iniziali = [40, 60, 80, 100, 120, 150]
    istanti_grafico_target = [120]  

    errori = {
        'willems': {'ou': [], 'ct_positive': [], 'ct_negative': [], 'switch': []},
        'svd': {'ou': [], 'ct_positive': [], 'ct_negative': [], 'switch': []}
    }

    dati_grafici = {}

    for i_test in range(800, 1000):
        Ti_test = lengths[i_test]
        maschera_test = mask[i_test].astype(bool)
        if np.sum(maschera_test) != Ti_test:
            raise ValueError(f"Incoerenza mask/lengths traiettoria test {i_test}")

        traiettoria_test_y = Y[i_test, :Ti_test, :]
        traiettoria_test_x = X[i_test, :Ti_test, :]
        modelli_test = active_model_ids[i_test, :Ti_test]

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

            modelli_finestra_passato = modelli_test[ist_ini: ist_ini + N_passato]
            tipo_finestra = classifica_finestra(modelli_finestra_passato)

            g_willems = H_Y_pinv_willems @ nuovo_Y_vec
            pred_willems = (H_Z @ g_willems).reshape(N_futuro, dim_misura)
            mse_willems = np.mean((futuro_reale_x - pred_willems) ** 2)

            g_svd = H_Y_pinv_svd @ nuovo_Y_vec
            pred_svd = (H_Z @ g_svd).reshape(N_futuro, dim_misura)
            mse_svd = np.mean((futuro_reale_x - pred_svd) ** 2)

            errori['willems'][tipo_finestra].append(mse_willems)
            errori['svd'][tipo_finestra].append(mse_svd)

            # Salva i dati per il grafico visivo se traiettoria = 850 e istante target t = 120
            if i_test == 850 and ist_ini in istanti_grafico_target:
                dati_grafici[ist_ini] = {
                    'passato': passato_osservato,
                    'futuro_x': futuro_reale_x,
                    'futuro_y': futuro_reale_y,
                    'pred_willems': pred_willems,
                    'pred_svd': pred_svd,
                    'tipo_finestra': tipo_finestra
                }

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Dataset: {etichetta_rumore}")
    print("=" * 70)
    print(f"Condition number H_Y: {condition_number:.2e}")
    print(f"Rango SVD: {rango_svd}")
    print()
    print(f"{'Tipo finestra':<15} {'N campioni':<12} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 57)

    tutti_w, tutti_s = [], []
    for tipo in ['ou', 'ct_positive', 'ct_negative', 'switch']:
        n = len(errori['willems'][tipo])
        if n > 0:
            rw = np.sqrt(np.mean(errori['willems'][tipo]))
            rs = np.sqrt(np.mean(errori['svd'][tipo]))
            print(f"{tipo:<15} {n:<12} {rw:<15.4f} {rs:<15.4f}")
            tutti_w.extend(errori['willems'][tipo])
            tutti_s.extend(errori['svd'][tipo])
        else:
            print(f"{tipo:<15} {'N/A':<12} {'N/A':<15} {'N/A':<15}")

    rmse_glob_w = np.sqrt(np.mean(tutti_w)) if tutti_w else float('nan')
    rmse_glob_s = np.sqrt(np.mean(tutti_s)) if tutti_s else float('nan')

    print("-" * 57)
    print(f"{'GLOBALE':<15} {len(tutti_w):<12} {rmse_glob_w:<15.4f} {rmse_glob_s:<15.4f}")

    # ---------------------------------------------------------
    # GRAFICO QUALITATIVO (SOLO t = 120)
    # ---------------------------------------------------------
    if 120 in dati_grafici:
        d = dati_grafici[120]
        plt.figure(figsize=(8, 6))
        plt.plot(d['passato'][:, 0], d['passato'][:, 1], "bo-", alpha=0.5, label="Passato Y")
        plt.plot(d['futuro_y'][:, 0], d['futuro_y'][:, 1], "go-", alpha=0.5, label="Futuro Y (rumoroso)")
        plt.plot(d['futuro_x'][:, 0], d['futuro_x'][:, 1], "k-", linewidth=2.5, label="Futuro X (vero)")
        plt.plot(d['pred_willems'][:, 0], d['pred_willems'][:, 1], "mX--", linewidth=1.5, label="Willems Std")
        plt.plot(d['pred_svd'][:, 0], d['pred_svd'][:, 1], "rD-.", linewidth=2, label=f"SVD (r={rango_svd})")

        plt.title(f"Confronto Predizione (Traiettoria 850, t = 120)\nRegime: {d['tipo_finestra']} - {etichetta_rumore}", fontsize=12)
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


# ==============================================================================
# ELENCO DATASET
# ==============================================================================
database_list = [
    ("data/trajectory_dataset_switching_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_0m_dt_1.npz", "std = 0 m (Noiseless)", 18),
    ("data/trajectory_dataset_switching_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_3m_dt_1.npz", "std = 3 m", 8),
    ("data/trajectory_dataset_switching_three_models_ou_ctpos_ctneg_ou_vx_m10_10_vy_m10_10_ct_3deg_variable_std_5m_dt_1.npz", "std = 5 m", 8),
]

risultati_globali = []
for percorso, etichetta, rango in database_list:
    try:
        risultato = confronto_switching3(percorso, etichetta, rango)
        risultati_globali.append(risultato)
    except FileNotFoundError:
        print(f"\n[ERRORE] File non trovato: {percorso} (verifica il nome esatto con ls data/)")

if risultati_globali:
    print("\n")
    print("=" * 90)
    print("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS SVD (SWITCHING 3 MODELLI)")
    print("=" * 90)
    print(f"{'Dataset':<22} {'Cond. Num.':<15} {'Rango':<8} {'RMSE Willems':<15} {'RMSE SVD':<15}")
    print("-" * 90)
    for r in risultati_globali:
        print(f"{r['etichetta']:<22} {r['condition_number']:<15.2e} {r['rango_svd']:<8} "
              f"{r['rmse_willems']:<15.4f} {r['rmse_svd']:<15.4f}")
    print("-" * 90)

    # --- CREAZIONE TABELLA MATPLOTLIB ---
    fig_tbl, ax_tbl = plt.subplots(figsize=(10, 3.5))
    ax_tbl.axis('off')

    intestazioni = ["Dataset", "Condition Number", "Rango SVD", "RMSE Willems [m]", "RMSE SVD [m]"]
    dati_celle = []
    for r in risultati_globali:
        dati_celle.append([
            r['etichetta'],
            f"{r['condition_number']:.2e}",
            str(r['rango_svd']),
            f"{r['rmse_willems']:.4f}",
            f"{r['rmse_svd']:.4f}"
        ])

    tabella = ax_tbl.table(
        cellText=dati_celle,
        colLabels=intestazioni,
        loc='center',
        cellLoc='center'
    )

    tabella.auto_set_font_size(False)
    tabella.set_fontsize(11)
    tabella.scale(1.2, 1.8)

    for (i, j), cell in tabella.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', weight='bold')
        else:
            colore_bg = '#f2f4f4' if i % 2 == 0 else '#ffffff'
            cell.set_facecolor(colore_bg)

    plt.title("TABELLA RIASSUNTIVA - CONFRONTO WILLEMS VS SVD (SWITCHING 3 MODELLI)", 
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()

plt.show()
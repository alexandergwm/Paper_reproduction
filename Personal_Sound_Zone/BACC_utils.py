import numpy as np

def build_Hml(h_ml, J):
    """
    构造卷积矩阵 H_ml
    参数:
    h_ml - RIR向量，长度K
    J - 滤波器长度
    """
    K = h_ml.shape[-1]
    N = K + J - 1
    H = np.zeros((N,J))
    for j in range(J):
        H[j:j+K, j] = h_ml
    return H

def test_Hml():
    MB = 2
    L = 2
    K = 10
    J = 5
    N = K + J - 1
    h_ml = np.random.randn(K)
    J = 5
    H = build_Hml(h_ml, J)
    print("H_ml矩阵: \n", H)

def get_rBk_n(HB, mic_idx, n, L, M, MB):
    """
    提取第mic_idx 个麦克风在时间点 n 的响应向量 rBk(n)
    参数:
    HB  - 全局传递函数 [MB*N x LJ]
    mic_idx - 麦克风索引(0<= mic_idx < MB)
    n - 时间点(0<= n <N)
    L - 扬声器数量
    M - 滤波器长度
    """
    N = HB.shape[0] // MB
    row_in_HB = mic_idx * N + n
    rBk_n = np.zeros(L * M)

    for l in range(L):
        col_start = l * M
        col_end = (l+1) * M
        rBk_n[col_start:col_end] = HB[row_in_HB, col_start:col_end]
    
    return rBk_n.reshape(-1, 1)

def build_all_rBk(HB, MB, L, M):
    N = HB.shape[0] // MB
    rBk = []
    for mic_idx in range(MB):
        mic_rBk = []
        for n in range(N):
            rBk_n = get_rBk_n(HB, mic_idx, n, L, M, MB)
            mic_rBk.append(rBk_n)
        rBk.append(mic_rBk)
    
    return rBk

def compute_sBk(freq_idx, HB, mic_idx, L, M, K, fs, MB):
    """
    计算第mic_idx个麦克风在频率freq_idx处的sBk(f)
    参数:
    freq_idx - 频率索引
    HB - 全局传递矩阵 [MB * N x LJ]
    mic_idx
    L, M ,K
    fs
    """
    N = K + M -1
    Ts = 1/fs
    n_values = np.arange(N)
    f = freq_idx * fs/ N
    exp_part = np.exp(-1j*2*np.pi*f*Ts*n_values)
    sBk = np.zeros(L*M, dtype=np.complex128)
    for n in range(N):
        rBk_n = get_rBk_n(HB, mic_idx, n, L, M, MB)
        sBk += rBk_n.flatten() * exp_part[n]

    return sBk.reshape(-1, 1)

def build_RD(HB, L, M, K, MB, fs, J_freqs):
    N = K+M-1
    Ts = 1/fs
    delta_f = fs/N

    V = np.zeros((J_freqs-1, J_freqs))
    for i in range(J_freqs- 1):
        V[i,i] = -1
        V[i,i+1] = 1
    V = V / delta_f

    Sk_list = []
    for k in range(MB):
        Sk = np.zeros((J_freqs, L*M), dtype=np.complex128)
        for j in range(J_freqs):
            freq_idx = j
            sBk = compute_sBk(freq_idx, HB, mic_idx=k, L=L, M=M, K=K, fs=fs, MB=MB)
            Sk[j,:] = sBk.flatten()
        Sk_list.append(Sk)

    RD = np.zeros((L*M, L*M), dtype=np.complex128)
    for k in range(MB):
        VSk = V @ Sk_list[k]
        RD += np.real(VSk.conj().T @ VSk)

    RD = RD / ((J_freqs-1)*MB)
    return RD
def build_RTE(HB, L, M, K, MB, fs, J_freqs, a):
    N = K+M-1
    delta_f = fs/N
    C0 = ((J_freqs - 2*a +1)*MB)
    Va = create_Va(a,J_freqs)
    Va = Va/(a**2*delta_f)

    Sk_list = []
    for k in range(MB):
        Sk = np.zeros((J_freqs, L*M), dtype=np.complex128)
        for j in range(J_freqs):
            freq_idx = j
            sBk = compute_sBk(freq_idx, HB, mic_idx=k, L=L, M=M, K=K, fs=fs, MB=MB)
            Sk[j,:] = sBk.flatten()
        Sk_list.append(Sk)

    RTE = np.zeros((L*M,L*M),dtype=np.complex128)
    for k in range(MB):
        VSk = Va@Sk_list[k]
        RTE += np.real(VSk.conj().T @VSk)
    RTE = RTE/C0
    return RTE

def create_Va(a, J):
    rows = (J - 2*a + 1)
    if rows <= 0:
        return []
    matrix = []
    for i in range(rows):
        row = [0] * J
        for j in range(i, i+1):
            row[j] = -1
        for j in range(i+a, i+2*a):
            row[j] = 1
        matrix.append(row)
    matrix = np.array(matrix)
    return matrix
def build_H(control_point, speaker_num, N, RIR, filter_len, shift_point=0):
    Mic_num = control_point
    Spk_num = speaker_num
    H_rows = Mic_num * N
    H_cols = Spk_num * filter_len
    H = np.zeros((H_rows, H_cols))
    for m in range(Mic_num):
        row_start = m * N
        row_end = (m+1) * N

        for l in range(Spk_num):
            col_start = l * filter_len
            col_end = (l+1) * filter_len
            h_ml = RIR[m+shift_point][l]
            H_ml = build_Hml(h_ml, filter_len)
            H[row_start:row_end, col_start:col_end] = H_ml
    return H

def solve_BACC(RB, RD, RD_term, beta = 0.5, delta=1e-5):
    matrix = np.linalg.pinv(beta*RD + (1-beta)*RD_term + delta * np.eye(np.size(RD.shape[0]), dtype=complex)) @ RB
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[max_eigenvalue_index]
    corresponding_eigenvector = eigenvectors[:, max_eigenvalue_index]

    w_opt = corresponding_eigenvector

    eB = w_opt.T @ RB @ w_opt
    eD = w_opt.T @ RD @ w_opt
    contrast = 10*np.log10(eB / (eD +delta))

    return w_opt, contrast


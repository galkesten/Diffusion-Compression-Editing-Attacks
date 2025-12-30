import math
import pickle
import os

class BitStreamEncoder:
    def __init__(self, K: int, M: int, C: int, device="cuda", use_precomputed_combs=True, to_base_path='.'):
        self.K = K
        self.M = M
        self.C = C

        print(f"[BitStreamEncoder] Computing bits_for_rank: comb({K}, {M})...")
        try:
            self.bits_for_rank = math.ceil(math.log2(math.comb(K, M)))
            print(f"[BitStreamEncoder] bits_for_rank = {self.bits_for_rank} bits")
        except (OverflowError, ValueError) as e:
            print(f"[BitStreamEncoder] ERROR: Cannot compute comb({K}, {M}) - {e}")
            print(f"[BitStreamEncoder] M={M} is too large! Try a smaller M value (e.g., M <= 50)")
            raise ValueError(f"M={M} is too large. The combination comb({K}, {M}) cannot be computed.")
        self.bits_for_coeff = self.C # bits for one coeff
        
        self.device = device
        self.accumulated_bitstring = ''
        self.use_precomputed_combs = use_precomputed_combs
        self.combs_table = None

        if use_precomputed_combs:
            full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
            if not os.path.exists(full_path):
                print(f"[BitStreamEncoder] Precomputed table not found for K={K}, M={M}")
                print(f"[BitStreamEncoder] Computing combination table... This may take a while for large M values.")
                print(f"[BitStreamEncoder] Table size: {K} x {M} = {K * M:,} entries")
                BitStreamEncoder.precompute_comb_table(K, M, C, to_base_path)
                print(f"[BitStreamEncoder] Combination table computed and saved to {full_path}")
            else:
                print(f"[BitStreamEncoder] Loading precomputed table from {full_path}")
            with open(full_path, "rb") as f:
                self.combs_table = pickle.load(f)['table']
                print(f"[BitStreamEncoder] Table loaded successfully ({len(self.combs_table)} x {len(self.combs_table[0]) if self.combs_table else 0})")

    def _compute_rank(self, comb: list[int]) -> int:
        rank = 0
        prev = -1

        if self.use_precomputed_combs:
            for i in range(self.M):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                    rank += self.combs_table[x][i]
                prev = c_i

        else:
            for i in range(self.M):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                        rank += math.comb(self.K - x - 1, self.M - i - 1)
                prev = c_i
            
        return rank


    def add(self, comb: list[int], coeffs: list[int]):
        # sorting the comb to compute lex order and coeffs accordingly
        sorted_pairs = sorted(zip(comb, coeffs))
        sorted_comb, sorted_coeffs = zip(*sorted_pairs)
        sorted_comb = list(sorted_comb)
        sorted_coeffs = list(sorted_coeffs)

        rank = self._compute_rank(sorted_comb)
        
        iteration_bitstring = format(rank, f'0{self.bits_for_rank}b') + ''.join(format(coeff, f'0{self.bits_for_coeff}b') for coeff in sorted_coeffs)
        self.accumulated_bitstring += iteration_bitstring
        
        return iteration_bitstring

    def get_encoding(self) -> list[int]:
        return self.accumulated_bitstring

    def clear(self):
        self.accumulated_bitstring = ''
    
    def decode(self, encoding):
        expected_bits_per_iteration = self.bits_for_rank + self.bits_for_coeff * self.M
        assert expected_bits_per_iteration > 8 # otherwise there might be confusion with the padding to bytes
        
        len_without_padding = expected_bits_per_iteration * (len(encoding) // expected_bits_per_iteration)
        encoding = encoding[len(encoding) - len_without_padding:]
        encoding_iterations_split = [encoding[i:i+expected_bits_per_iteration] for i in range(0, len(encoding), expected_bits_per_iteration)]
        
        decoding = []
        for iteration_bits in encoding_iterations_split:
            rank = int(iteration_bits[:self.bits_for_rank], 2)
            coeffs = iteration_bits[self.bits_for_rank:]
            coeffs = [int(coeffs[i:i+self.bits_for_coeff], 2) for i in range(0, len(coeffs), self.bits_for_coeff)]

            # decode combination
            comb = []
            x = 0
            for i in range(self.M):
                while True:
                    value = self.combs_table[x][i] if self.use_precomputed_combs else math.comb(self.K - x - 1, self.M - i - 1)
                    if value > rank:
                        break
                    rank -= value
                    x += 1
                comb.append(x)
                x += 1
            
            decoding.append((comb, coeffs))
        
        return decoding

    def get_pkl_full_path(K, M, to_base=''):
        pkl_name = f"comb_K{K}_M{M}.pkl"
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "combs_files", pkl_name)

    @staticmethod
    def precompute_comb_table(K: int, M: int, C, to_base_path='.'):
        full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
        print(f"[Precompute] Creating table of size {K} x {M} = {K * M:,} entries...")
        table = [[0 for i in range(M)] for x in range(K)]
        
        total_entries = K * M
        entries_computed = 0
        last_progress = 0
    
        for x in range(K):
            if x % 100 == 0 or x == K - 1:  # Print progress every 100 rows
                progress = int((entries_computed / total_entries) * 100)
                if progress != last_progress:
                    print(f"[Precompute] Progress: {progress}% ({entries_computed:,}/{total_entries:,} entries)")
                    last_progress = progress
            for i in range(M):
                n = K - x - 1
                k = M - i - 1
                if n >= 0 and k >= 0 and n >= k:
                    try:
                        table[x][i] = math.comb(n, k)
                    except (OverflowError, ValueError) as e:
                        print(f"[Precompute] ERROR at x={x}, i={i}: {e}")
                        print(f"[Precompute] Attempting to compute comb({n}, {k}) - number too large!")
                        raise
                else:
                    table[x][i] = 0
                entries_computed += 1
    
        print(f"[Precompute] Saving table to {full_path}...")
        with open(full_path, 'wb') as f:
            pickle.dump({
                'K': K,
                'M': M,
                'table': table
            }, f)
        print(f"[Precompute] Table saved successfully!")
    
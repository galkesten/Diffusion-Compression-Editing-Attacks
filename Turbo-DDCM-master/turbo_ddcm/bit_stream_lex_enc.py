import math
import pickle
import os

class BitStreamEncoder:
    def __init__(self, K: int, M: int, C: int, old_protocol_ind, device="cuda", use_precomputed_combs=True, to_base_path='.'):
        self.K = K
        self.M = M
        self.C = C
        
        self.old_protocol_ind = old_protocol_ind
        
        self.bits_for_coeff = self.C # bits for one coeff
        self.device = device
        self.accumulated_bitstring = ''
        
        if not self.old_protocol_ind:
            self.bits_for_rank = math.ceil(math.log2(math.comb(K, M)))
            self.use_precomputed_combs = use_precomputed_combs
            self.combs_table = None
    
            if use_precomputed_combs:
                full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
                if not os.path.exists(full_path):
                    print(f"Combination table not found at {full_path}, computing...")
                    BitStreamEncoder.precompute_comb_table(K, M, C, to_base_path)
                else:
                    print(f"Loading combination table from {full_path}")
                with open(full_path, "rb") as f:
                    self.combs_table = pickle.load(f)['table']
                    print(f"Combination table loaded successfully")

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
        sorted_pairs = sorted(zip(comb, coeffs))
        sorted_comb, sorted_coeffs = zip(*sorted_pairs)
        sorted_comb = list(sorted_comb)
        sorted_coeffs = list(sorted_coeffs)
        
        if self.old_protocol_ind:
            bits_per_choice = math.ceil(math.log2(self.K))
            bitstream = ''.join(format(i, f'0{bits_per_choice}b') for i in sorted_comb)
            iteration_bitstring = bitstream + ''.join(format(coeff, f'0{self.bits_for_coeff}b') for coeff in sorted_coeffs)
            
        else:
            # sorting the comb to compute lex order and coeffs accordingly    
            rank = self._compute_rank(sorted_comb)
            iteration_bitstring = format(rank, f'0{self.bits_for_rank}b') + ''.join(format(coeff, f'0{self.bits_for_coeff}b') for coeff in sorted_coeffs)
            
        self.accumulated_bitstring += iteration_bitstring
        return iteration_bitstring

    def get_encoding(self):
        return self.accumulated_bitstring

    def clear(self):
        self.accumulated_bitstring = ''
    
    def decode(self, encoding):
        expected_bits_per_iteration = (math.ceil(math.log2(self.K)) + self.bits_for_coeff) * self.M if self.old_protocol_ind else self.bits_for_rank + self.bits_for_coeff * self.M
        assert expected_bits_per_iteration > 8 # otherwise there might be confusion with the padding to bytes
        
        len_without_padding = expected_bits_per_iteration * (len(encoding) // expected_bits_per_iteration)
        encoding = encoding[len(encoding) - len_without_padding:]
        encoding_iterations_split = [encoding[i:i+expected_bits_per_iteration] for i in range(0, len(encoding), expected_bits_per_iteration)]
        
        decoding = []
        for iteration_bits in encoding_iterations_split:
            comb = []
            if self.old_protocol_ind:
                bits_per_choice = math.ceil(math.log2(self.K))
                comb = [int(iteration_bits[i:i+bits_per_choice], 2) for i in range(0, bits_per_choice*self.M, bits_per_choice)]
                coeffs = iteration_bits[math.ceil(math.log2(self.K)) * self.M:]
                coeffs = [int(coeffs[i:i+self.bits_for_coeff], 2) for i in range(0, len(coeffs), self.bits_for_coeff)]

            else:
                rank = int(iteration_bits[:self.bits_for_rank], 2)
                coeffs = iteration_bits[self.bits_for_rank:]
                coeffs = [int(coeffs[i:i+self.bits_for_coeff], 2) for i in range(0, len(coeffs), self.bits_for_coeff)]
    
                # decode combination
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

    @staticmethod
    def get_pkl_full_path(K, M, to_base=''):
        pkl_name = f"comb_K{K}_M{M}.pkl"
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "combs_files", pkl_name)

    @staticmethod
    def precompute_comb_table(K: int, M: int, C, to_base_path='.'):
        full_path = BitStreamEncoder.get_pkl_full_path(K, M, to_base_path)
        combs_dir = os.path.dirname(full_path)
        os.makedirs(combs_dir, exist_ok=True)
        print(f"Computing combination table for K={K}, M={M}")
        print(f"Saving to: {full_path}")
        
        table = [[0 for i in range(M)] for x in range(K)]
    
        for x in range(K):
            for i in range(M):
                n = K - x - 1
                k = M - i - 1
                if n >= 0 and k >= 0 and n >= k:
                    table[x][i] = math.comb(n, k)
                else:
                    table[x][i] = 0
    
        with open(full_path, 'wb') as f:
            pickle.dump({
                'K': K,
                'M': M,
                'table': table
            }, f)
        print(f"Combination table saved successfully to {full_path}")
    
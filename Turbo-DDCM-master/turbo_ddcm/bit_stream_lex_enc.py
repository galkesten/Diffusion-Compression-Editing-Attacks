import math
import pickle
import os

class BitStreamEncoder:
    def __init__(self, K: int, M: int, C: int, B: int, device="cuda", use_precomputed_combs=True, to_base_path='.'):
        self.K = K
        self.M = M
        self.B = B
        self.MmB = M - B
        self.C = C
        
        self.bits_for_coeff = self.C # bits for one coeff
        self.device = device
        self.accumulated_bitstring = ''

        self.bits_for_rank = math.ceil(math.log2(math.comb(K - self.B, self.MmB)))
        if self.MmB > 0:
            self.use_precomputed_combs = use_precomputed_combs
            self.combs_table = None

            if use_precomputed_combs:
                full_path = BitStreamEncoder.get_pkl_full_path(K - self.B, self.MmB, to_base_path)
                if not os.path.exists(full_path):
                    print(f"Combination table not found at {full_path}, computing...")
                    BitStreamEncoder.precompute_comb_table(K - self.B, self.MmB, C, to_base_path)
                else:
                    print(f"Loading combination table from {full_path}")
                with open(full_path, "rb") as f:
                    self.combs_table = pickle.load(f)['table']
                    print(f"Combination table loaded successfully")

    def _compute_rank(self, comb: list[int]) -> int:
        rank = 0
        prev = -1

        if self.use_precomputed_combs:
            for i in range(self.MmB):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                    rank += self.combs_table[x][i]
                prev = c_i

        else:
            for i in range(self.MmB):
                start = prev + 1
                c_i = comb[i]
                for x in range(start, c_i):
                        rank += math.comb(self.K - x - 1, self.MmB - i - 1)
                prev = c_i
            
        return rank


    def add(self, comb: list[int], coeffs: list[int]):
        iteration_bitstring = ''
        coeffs_out = []

        if self.B > 0: # old protocol
            bits_per_choice = math.ceil(math.log2(self.K))
            iteration_bitstring += ''.join(format(i, f'0{bits_per_choice}b') for i in comb[:self.B])
            coeffs_out += coeffs[:self.B]

        if self.MmB > 0: # new protocol
            sorted_pairs = sorted(zip(comb[self.B:], coeffs[self.B:]))
            sorted_comb, sorted_coeffs = zip(*sorted_pairs)
            sorted_comb = list(sorted_comb)
            sorted_coeffs = list(sorted_coeffs)

            cs_ = ComplementSet(self.K, comb[:self.B])
            mapped_sorted_comb = [cs_.get_index(num) for num in sorted_comb]

            # sorting the comb to compute lex order and coeffs accordingly
            rank = self._compute_rank(mapped_sorted_comb)
            iteration_bitstring += format(rank, f'0{self.bits_for_rank}b')
            coeffs_out += sorted_coeffs

        iteration_bitstring += ''.join(format(coeff, f'0{self.bits_for_coeff}b') for coeff in coeffs_out)
        self.accumulated_bitstring += iteration_bitstring
        return iteration_bitstring

    def get_encoding(self):
        return self.accumulated_bitstring

    def clear(self):
        self.accumulated_bitstring = ''
    
    def decode(self, encoding):
        expected_bits_per_iteration = self.bits_for_rank + math.ceil(math.log2(self.K)) * self.B + self.bits_for_coeff * self.M
        assert expected_bits_per_iteration > 8 # otherwise there might be confusion with the padding to bytes
        
        len_without_padding = expected_bits_per_iteration * (len(encoding) // expected_bits_per_iteration)
        encoding = encoding[len(encoding) - len_without_padding:]
        encoding_iterations_split = [encoding[i:i+expected_bits_per_iteration] for i in range(0, len(encoding), expected_bits_per_iteration)]
        decoding = []
        for iteration_bits in encoding_iterations_split:
            comb = []
            passed_bits = 0
            if self.B > 0:
                bits_per_choice = math.ceil(math.log2(self.K))
                comb = [int(iteration_bits[i:i+bits_per_choice], 2) for i in range(0, bits_per_choice*self.B, bits_per_choice)]
                passed_bits += bits_per_choice * self.B

            if self.MmB > 0:
                temp_comb = []
                rank = int(iteration_bits[passed_bits:passed_bits+self.bits_for_rank], 2)
                # decode combination
                x = 0
                for i in range(self.MmB):
                    while True:
                        value = self.combs_table[x][i] if self.use_precomputed_combs else math.comb( (self.K - self.B) - x - 1, self.MmB - i - 1)
                        if value > rank:
                            break
                        rank -= value
                        x += 1
                    temp_comb.append(x)
                    x += 1

                cs_ = ComplementSet(self.K, comb.copy())
                comb += [cs_.get_by_index(index) for index in temp_comb]

                passed_bits += self.bits_for_rank

            coeffs = iteration_bits[passed_bits:]
            coeffs = [int(coeffs[i:i+self.bits_for_coeff], 2) for i in range(0, len(coeffs), self.bits_for_coeff)]

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



import bisect
class ComplementSet:
    def __init__(self, K, A):
        """
        K: upper bound (inclusive)
        A: iterable of unique integers in [1, K]
        """
        self.K = K
        self.A = sorted(set(A))

    def _count_removed_leq(self, x):
        """Number of elements in A <= x"""
        return bisect.bisect_right(self.A, x)

    def size(self):
        """Number of elements in the complement set"""
        return self.K - len(self.A)

    def get_by_index(self, i):
        """
        Returns the i-th smallest number in [1..K] excluding A.
        Raises IndexError if out of range.
        """
        i += 1 # to 1 base
        if i < 1 or i > self.size():
            raise IndexError("Index out of range")

        low, high = 1, self.K

        while low <= high:
            mid = (low + high) // 2
            removed = self._count_removed_leq(mid)
            missing = mid - removed

            if missing < i:
                low = mid + 1
            else:
                high = mid - 1

        return low

    def get_index(self, x):
        """
        Returns the 1-based index of x in the complement set.
        Returns -1 if x is removed or out of range.
        """
        if x < 1 or x > self.K:
            return -1

        pos = bisect.bisect_left(self.A, x)
        if pos < len(self.A) and self.A[pos] == x:
            return -1  # x is excluded

        return x - pos - 1 # to zero base

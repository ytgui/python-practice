import hashlib
import binascii


class Secp256k1:
    G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
         0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
    M = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F

    def __init__(self):
        pass

    def is_not_used(self):
        pass

    def mod_inverse(self, x):
        inv1, inv2 = 1, 0
        p = self.M
        while p != 1 and p != 0:
            inv1, inv2 = inv2, inv1 - inv2 * (x // p)
            x, p = p, x % p
        if p == 0:
            return None
        return inv2

    def pt_dbl(self, p1):
        if p1 is None:
            return None
        x, y = p1
        if y == 0:
            return None

        # Calculate 3*x^2/(2*y)  modulus p
        slope = 3 * pow(x, 2, self.M) * self.mod_inverse(2 * y)
        x_sum = pow(slope, 2, self.M) - 2 * x
        y_sum = slope * (x - x_sum) - y

        return x_sum % self.M, y_sum % self.M

    def pt_add(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            return self.pt_dbl(p1)

        # calculate (y1-y2)/(x1-x2)  modulus p
        slope = (y1 - y2) * self.mod_inverse(x1 - x2)
        x_sum = pow(slope, 2, self.M) - (x1 + x2)
        y_sum = slope * (x1 - x_sum) - y1
        return x_sum % self.M, y_sum % self.M

    def pt_mul(self, a, pt):
        scale = pt
        acc = None
        while a:
            if a & 1:
                if acc is None:
                    acc = scale
                else:
                    acc = self.pt_add(acc, scale)
            scale = self.pt_dbl(scale)
            a >>= 1
        return acc

    def is_on_curve(self, pt):
        x, y = pt
        return (y ** 2 - x ** 3 - 7) % self.M == 0

    def generate_pubkey(self, private_key):
        public_key = self.pt_mul(private_key, self.G)
        assert self.is_on_curve(public_key)
        return public_key


def test_ecc():
    private_key = 0xf8ef380d6c05116dbed78bfdd6e6625e57426af9a082b81c2fa27b06984c11f3
    public_key = Secp256k1().generate_pubkey(private_key)
    assert public_key == (0x71ee918bc19bb566e3a5f12c0cd0de620bec1025da6e98951355ebbde8727be3,
                          0x37b3650efad4190b7328b1156304f2e9e23dbb7f2da50999dde50ea73b4c2688)


class Base58:
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    base_count = len(alphabet)

    def encode(self, num):
        """ Returns num in a base58-encoded string """
        encode = ''

        if num < 0:
            return ''

        while num >= self.base_count:
            mod = num % self.base_count
            encode = self.alphabet[mod] + encode
            num = num // self.base_count

        if num:
            encode = self.alphabet[num] + encode

        return encode

    def decode(self, s):
        """ Decodes the base58-encoded string s into an integer """
        decoded = 0
        multi = 1
        s = s[::-1]
        for char in s:
            decoded += multi * self.alphabet.index(char)
            multi = multi * self.base_count

        return decoded


class BtcAddress:
    version = 0
    prefix = '1'

    def base58_check(self, src):
        src = bytes([self.version]) + src
        h = hashlib.sha256()
        h.update(src)
        r = h.digest()

        h = hashlib.sha256()
        h.update(r)
        r = h.digest()

        checksum = r[:4]
        s = src + checksum

        return Base58().encode(int.from_bytes(s, "big"))

    def generate_address(self, public_key):
        x, y = public_key
        s = b'\x04' + x.to_bytes(length=32, byteorder='big') + y.to_bytes(length=32, byteorder='big')

        h = hashlib.sha256()
        h.update(s)
        r = h.digest()

        h = hashlib.new("ripemd160")
        h.update(r)
        r = h.digest()

        return self.prefix + '{}'.format(self.base58_check(r))


def test_address():
    public_key = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                  0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
    assert BtcAddress().generate_address(public_key) == '1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm'


def main():
    test_ecc()
    test_address()


if __name__ == '__main__':
    main()

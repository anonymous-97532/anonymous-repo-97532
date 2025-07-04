// Package sampling implements secure sampling of bytes and integers.
package sampling

import (
	"crypto/rand"
	"encoding/binary"
	"math/big"
)

// RandUint64 return a random value between 0 and 0xFFFFFFFFFFFFFFFF.
func RandUint64() uint64 {
	b := []byte{0, 0, 0, 0, 0, 0, 0, 0}
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return binary.LittleEndian.Uint64(b)
}

// RandFloat64 returns a random float between min and max.
func RandFloat64(min, max float64) float64 {
	b := []byte{0, 0, 0, 0, 0, 0, 0, 0}
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	f := float64(binary.LittleEndian.Uint64(b)) / 1.8446744073709552e+19
	return min + f*(max-min)
}

// RandComplex128 returns a random complex with the real and imaginary part between min and max.
func RandComplex128(min, max float64) complex128 {
	return complex(RandFloat64(min, max), RandFloat64(min, max))
}

func RandFloat128(min, max float64) complex128 {
	return complex(RandFloat64(min, max), 0)
}

// RandInt generates a random Int in [0, max-1].
func RandInt(max *big.Int) (n *big.Int) {
	var err error
	if n, err = rand.Int(rand.Reader, max); err != nil {
		panic(err)
	}
	return
}

func RandModFloat(mod uint64) complex128 {
    return complex(float64(RandMod(mod)) / float64(mod), 0)
}

func RandMod(mod uint64) uint64 {
    if mod == 0 {
        panic("mod must be greater than 0")
    }

    maxAcceptable := ^uint64(0) - (^uint64(0) % mod)

    for {
        r := RandUint64()
        if r <= maxAcceptable {
	    return r % mod
        }
    }
}

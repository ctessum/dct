// Package dct implements forward and reverse discrete cosine fourier transforms.
// The algorithm is based on the information at:
// https://www.mathworks.com/help/images/ref/dct2.html and
// https://www.mathworks.com/help/images/ref/idct2.html.
package dct

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// FDCT computes the forward discrete cosine transform of src and places it in dst,
// also returning dst.
// If dst is nil, a new matrix is allocated and returned.
func FDCT(src, dst *mat.Dense) *mat.Dense {
	r, c := src.Dims()
	if r%2 != 0 || c%2 != 0 {
		panic(fmt.Errorf("dct: matrix dimensions must be even"))
	}
	if dst == nil {
		dst = mat.NewDense(r, c, nil)
	}
	N1 := float64(r)
	N2 := float64(c)
	for ik1 := 0; ik1 < r; ik1++ {
		k1 := float64(ik1)
		for ik2 := 0; ik2 < c; ik2++ {
			k2 := float64(ik2)
			var sum float64
			for in1 := 0; in1 < r; in1++ {
				n1 := float64(in1)
				for n2 := 0; n2 < c; n2++ {
					sum += src.At(in1, n2) * math.Cos(math.Pi/N1*(n1+0.5)*k1) * math.Cos(math.Pi/N2*(float64(n2)+0.5)*k2)
				}
			}
			var ck1, ck2 float64
			if ik1 == 0 {
				ck1 = 1 / math.Sqrt(N1)
			} else {
				ck1 = math.Sqrt(2 / N1)
			}
			if ik2 == 0 {
				ck2 = 1 / math.Sqrt(N2)
			} else {
				ck2 = math.Sqrt(2 / N2)
			}
			dst.Set(ik1, ik2, sum*ck1*ck2)
		}
	}
	return dst
}

// IDCT computes the inverse discrete cosine transform of src and places it in dst,
// also returning dst.
// If dst is nil, a new matrix is allocated and returned.
func IDCT(src, dst *mat.Dense) *mat.Dense {
	r, c := src.Dims()
	if r%2 != 0 || c%2 != 0 {
		panic(fmt.Errorf("dct: matrix dimensions must be even"))
	}
	if dst == nil {
		dst = mat.NewDense(r, c, nil)
	}
	N1 := float64(r)
	N2 := float64(c)
	for ik1 := 0; ik1 < r; ik1++ {
		k1 := float64(ik1)
		for ik2 := 0; ik2 < c; ik2++ {
			k2 := float64(ik2)
			var sum, cn1, cn2 float64
			for in1 := 0; in1 < r; in1++ {
				if in1 == 0 {
					cn1 = 1 / math.Sqrt(N1)
				} else {
					cn1 = math.Sqrt(2 / N1)
				}
				n1 := float64(in1)
				for n2 := 0; n2 < c; n2++ {
					if n2 == 0 {
						cn2 = 1 / math.Sqrt(N2)
					} else {
						cn2 = math.Sqrt(2 / N2)
					}
					sum += src.At(in1, n2) * cn1 * cn2 * math.Cos(math.Pi/N1*(k1+0.5)*n1) * math.Cos(math.Pi/N2*(k2+0.5)*float64(n2))
				}
			}
			dst.Set(ik1, ik2, sum)
		}
	}
	return dst
}

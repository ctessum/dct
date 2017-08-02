package dct

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestDCT(t *testing.T) {
	tests := []*mat.Dense{
		mat.NewDense(4, 4, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
		mat.NewDense(2, 2, []float64{0, 1, 0, 1}),
		mat.NewDense(2, 2, []float64{6, 1, 6, 1}),
		mat.NewDense(4, 2, []float64{0, 1, 2, 3, 4, 5, 6, 7}),
	}
	for i, x := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			dct := FDCT(x, nil)
			x2 := IDCT(dct, nil)
			r, c := x.Dims()
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					if different(x.At(i, j), x2.At(i, j), 1.0e-10) {
						t.Errorf("index (%d,%d), want %g but have %g", i, j, x.At(i, j), x2.At(i, j))
					}
				}
			}
		})
	}
}

func different(a, b, tolerance float64) bool {
	if math.Abs(a-b)/math.Abs(a+b)*2 > tolerance && math.Abs(a-b) > tolerance {
		return true
	}
	return false
}

// RUN: heir-opt --poly-ext-to-polynomial %s  | FileCheck %s

//TODO: Add FileCheck tests

!p = !polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 33538049 : i32, polynomialModulus = <1 + x**1024>>>

func.func @gadget_product(%arg0: !p, %arg1 : !p) -> !p {
    %0 = poly_ext.gadget_product %arg0, %arg1 : (!p, !p) -> !p
    return %0 : !p
}

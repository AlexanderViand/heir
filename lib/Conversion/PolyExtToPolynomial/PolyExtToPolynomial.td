#ifndef LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_TD_
#define LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_TD_

include "mlir/Pass/PassBase.td"

def PolyExtToPolynomial : Pass<"poly-ext-to-polynomial"> {
  let summary = "Lower `poly_ext` to `polynomial` dialect.";
  let dependentDialects = [
    "mlir::heir::poly_ext::PolyExtDialect",
    "mlir::polynomial::PolynomialDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYEXTTOPOLYNOMIAL_POLYEXTTOPOLYNOMIAL_TD_

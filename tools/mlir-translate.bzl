"""A rule for running mlir-translate."""

def executable_attr(label):
    """A helper for declaring executable dependencies."""
    return attr.label(
        default = Label(label),
        allow_single_file = True,
        executable = True,
        # commenting this out breaks cross-compilation, but this should not be a problem
        # for developer builds
        # cfg = "exec",
        cfg = "target",
    )

_MLIR_TRANSLATE = "@llvm-project//mlir:mlir-translate"

def _mlir_translate_impl(ctx):
    generated_file = ctx.outputs.generated_filename
    args = ctx.actions.args()
    args.add_all(ctx.attr.pass_flags)
    args.add_all(["-o", generated_file.path])
    args.add(ctx.file.src)

    ctx.actions.run(
        mnemonic = "MLIRTranslateRule",
        inputs = ctx.attr.src.files,
        outputs = [generated_file],
        arguments = [args],
        executable = ctx.executable._mlir_translate_binary,
        toolchain = None,
    )
    return [
        DefaultInfo(files = depset([generated_file, ctx.file.src])),
    ]

mlir_translate = rule(
    doc = """
      This rule takes MLIR input and runs mlir-translate on it to produce
      a single generated source file in some target language.
      """,
    implementation = _mlir_translate_impl,
    attrs = {
        "src": attr.label(
            doc = "A single MLIR source file to translate.",
            allow_single_file = [".mlir"],
        ),
        "pass_flags": attr.string_list(
            doc = """
            The pass flags passed to mlir-translate, e.g., --mlir-to-llvmir.
            """,
        ),
        "generated_filename": attr.output(
            doc = """
            The name used for the output file, including the extension (e.g.,
            <filename>.ll for LLVMIR files).
            """,
            mandatory = True,
        ),
        "_mlir_translate_binary": executable_attr(_MLIR_TRANSLATE),
    },
)

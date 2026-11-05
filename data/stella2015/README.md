# Stella (2015)

`source.zip` preserves the author's original release, identified in `source.json`.
The pipeline extracts it into `code/.cache/upstream/`. Our wrappers call the
original constructors and model code from outside that extraction.

`public_inputs/` contains raw BEA, Census and BLS sources and their provenance.
Run `python -m applications.stella.prepare_public` from `code/` to reconstruct
employment denominators and the author's deposited GDP vector under
`code/.cache/derived/stella/`.

The full reconstruction also needs the local Compustat WRDS revenue, annual and
customer extracts and the reviewed firm/sector and customer-name maps. The paper
runner leaves Stella's estimates blank when these licensed inputs are absent.

For the licensed reconstruction, build the bundled X-12 seasonal-adjustment
engine once with `python -m applications.stella.build_stella_x12` from `code/`
(requires GNU Fortran), or set `X12PATH` to an existing executable. Public-input
sources and conventions are described in `public_inputs/README.md`.

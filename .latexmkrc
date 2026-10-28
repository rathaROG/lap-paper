# Compile the supplied manuscript directly, from the repository root.
@default_files = ('latex/main.tex');
$pdf_mode = 1;
$do_cd = 1;
$jobname = 'native_rectangular_assignment';

# With $do_cd enabled, these paths are relative to latex/.
$out_dir = '../output/pdf';
$aux_dir = '../output/build';
$pdflatex = 'pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -file-line-error %O %S';

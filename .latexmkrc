# Compile the supplied manuscript directly, from the repository root.
@default_files = ('latex/main.tex');
$pdf_mode = 1;
$do_cd = 1;
# Name only the main manuscript specially. Other roots (including the atlas)
# keep their own basename, so a supplementary build cannot replace the paper.
my @sources = grep { !/^-/ && (/\.(tex|ltx)$/i || -f "$_.tex") } @ARGV;
$jobname = (!@sources || (@sources == 1 && $sources[0] =~ m{(?:^|[/\\])main(?:\.tex)?$}))
    ? 'native_rectangular_assignment' : '%A';

# With $do_cd enabled, these paths are relative to latex/.
$out_dir = '../output/pdf';
$aux_dir = '../output/build';
$pdflatex = 'pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -file-line-error %O %S';

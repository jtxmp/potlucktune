# Records the README demo GIF. From the repo root:
#   vhs demo/demo.tape
# Needs vhs and ttyd on PATH, plus everything demo/drive.sh needs.
Output demo/demo.gif

Set Shell "bash"
Set FontFamily "Source Code Pro"
Set FontSize 13
Set Width 1800
Set Height 760
Set Padding 12
Set Framerate 15
Set Theme "Builtin Dark"

Hide
Type "PS1='' bash demo/drive.sh" Enter
Sleep 2s
Show

Wait+Screen@400s /no tickets left/
Sleep 5s

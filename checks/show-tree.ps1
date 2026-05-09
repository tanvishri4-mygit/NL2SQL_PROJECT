function Show-Tree {
    param (
        [string]$Path = ".",
        [string]$Exclude = "data"
    )

    Get-ChildItem -Path $Path -Recurse -Force |
        Where-Object { $_.FullName -notmatch "\\$Exclude\\" } |
        ForEach-Object {
            $relativePath = $_.FullName.Replace((Resolve-Path $Path), "")
            $depth = ($relativePath -split "[\\\/]").Count - 1
            (" " * 2 * $depth) + "|-- " + $_.Name
        }
}

Show-Tree
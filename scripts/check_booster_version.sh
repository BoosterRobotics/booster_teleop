# Sourced by the onboard installer and teleoperation launcher.
require_booster_version() {
    local file=/opt/booster/version.txt minimum=1.8.4
    local line version='' reason i part required
    local pattern='^[[:space:]]*Version:[[:blank:]]*([0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)*)(-[^[:space:]]+)?[[:space:]]*$'
    local -a actual threshold

    if [[ -f "$file" && -r "$file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ "$line" =~ ^[[:space:]]*Version: ]] || continue
            version=''
            if [[ "$line" =~ $pattern ]]; then
                version=${BASH_REMATCH[1]}
            fi
        done < "$file"
        reason="missing or invalid final Version field in $file"
    else
        reason="cannot read $file"
    fi

    if [[ -n "$version" ]]; then
        IFS=. read -r -a actual <<< "$version"
        IFS=. read -r -a threshold <<< "$minimum"
        for ((i = 0; i < ${#actual[@]}; i++)); do
            part=${actual[i]}
            part=${part#"${part%%[!0]*}"} # Strip leading zeroes before arithmetic.
            part=${part:-0}
            required=${threshold[i]:-0}
            # All threshold components are one digit; avoid overflow on malformed large numbers.
            if (( ${#part} > 1 || part > required )); then
                return 0
            elif (( part < required )); then
                break
            fi
        done
        reason="found $version in $file"
    fi

    printf 'Error: Booster firmware must be newer than %s (%s). Upgrade the robot firmware before continuing.\n' \
        "$minimum" "$reason" >&2
    return 1
}

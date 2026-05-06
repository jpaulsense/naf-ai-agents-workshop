#!/bin/bash
#
# Manage student access to the workshop repo
#
# Usage:
#   ./scripts/manage_students.sh add students.txt
#   ./scripts/manage_students.sh remove students.txt
#   ./scripts/manage_students.sh list
#
# students.txt should have one GitHub username per line.

REPO="jpaulsense/naf-ai-agents-workshop"

usage() {
    echo "Usage:"
    echo "  $0 add <students.txt>     Add students as collaborators"
    echo "  $0 remove <students.txt>  Remove student collaborators"
    echo "  $0 list                   List current collaborators"
    echo ""
    echo "students.txt format: one GitHub username per line"
}

add_students() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi

    echo "Adding students to $REPO..."
    echo ""

    local added=0
    local failed=0

    while IFS= read -r username || [ -n "$username" ]; do
        # Skip empty lines and comments
        username=$(echo "$username" | xargs)
        [[ -z "$username" || "$username" == \#* ]] && continue

        if gh api "repos/$REPO/collaborators/$username" -X PUT -f permission=read --silent 2>/dev/null; then
            echo "  ✓ $username — invited"
            ((added++))
        else
            echo "  ✗ $username — FAILED (check username)"
            ((failed++))
        fi
    done < "$file"

    echo ""
    echo "Done: $added invited, $failed failed"
    echo "Students must accept the invite at https://github.com/notifications"
}

remove_students() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi

    echo "Removing students from $REPO..."
    echo ""

    local removed=0

    while IFS= read -r username || [ -n "$username" ]; do
        username=$(echo "$username" | xargs)
        [[ -z "$username" || "$username" == \#* ]] && continue

        if gh api "repos/$REPO/collaborators/$username" -X DELETE --silent 2>/dev/null; then
            echo "  ✓ $username — removed"
            ((removed++))
        else
            echo "  - $username — not a collaborator (skipped)"
        fi
    done < "$file"

    echo ""
    echo "Done: $removed removed"
}

list_collaborators() {
    echo "Collaborators on $REPO:"
    echo ""
    gh api "repos/$REPO/collaborators" --jq '.[] | "  \(.login) (\(.role_name))"' 2>/dev/null
    echo ""
}

# Main
case "${1:-}" in
    add)
        [ -z "${2:-}" ] && { usage; exit 1; }
        add_students "$2"
        ;;
    remove)
        [ -z "${2:-}" ] && { usage; exit 1; }
        remove_students "$2"
        ;;
    list)
        list_collaborators
        ;;
    *)
        usage
        exit 1
        ;;
esac

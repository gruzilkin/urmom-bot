def get_country_from_flag(emoji: str) -> str | None:
    # Common country flags mapping
    flag_to_country = {
        "🇺🇸": "America",
        "🇷🇺": "Soviet Russia",
        "🇨🇵": "France",
        "🇨🇦": "Canada",
        "🇬🇧": "Britain",
        "🇩🇪": "Germany",
        "🇮🇹": "Italy",
        "🇯🇵": "Japan",
        "🇨🇳": "China",
        "🇦🇺": "Australia",
        "🇧🇷": "Brazil",
        "🇨🇺": "Cuba"
    }

    if emoji not in flag_to_country:
        print(f'Unknown flag emoji: "{emoji}"')
    return flag_to_country.get(emoji)

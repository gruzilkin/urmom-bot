"""
Test double for UserResolver pre-populated with famous physicists.
This eliminates the need for mocking and provides realistic test data.
"""

from user_resolver import UserResolver


class TestUserResolver(UserResolver):
    """Test double UserResolver pre-populated with famous physicists."""
    
    def __init__(self):
        # Don't call super().__init__() since we don't need real Discord client
        self._physicists = {
            # User ID based on birth-death years
            187911955: "Einstein",     # Albert Einstein (1879-1955)
            185819470: "Planck",       # Max Planck (1858-1947) 
            188519620: "Bohr",         # Niels Bohr (1885-1962)
            185619400: "Thomson",      # J.J. Thomson (1856-1940)
            187119370: "Rutherford",   # Ernest Rutherford (1871-1937)
            188719610: "Schrödinger",  # Erwin Schrödinger (1887-1961)
            190119760: "Heisenberg",   # Werner Heisenberg (1901-1976)
            188219700: "Born",         # Max Born (1882-1970)
            186719340: "Curie",        # Marie Curie (1867-1934)
            185319280: "Lorentz",      # Hendrik Lorentz (1853-1928)
            186419090: "Minkowski",    # Hermann Minkowski (1864-1909)
            190219840: "Dirac",        # Paul Dirac (1902-1984)
            190019580: "Pauli",        # Wolfgang Pauli (1900-1958)
            189219870: "de_Broglie",   # Louis de Broglie (1892-1987)
        }
        
        # Pre-populate display name cache to avoid API calls
        self._display_name_cache = {}
        for user_id, name in self._physicists.items():
            # Cache for all possible guild IDs used in tests
            for guild_id in [19001930, 12345]:  # Physics guild + fallback
                self._display_name_cache[(guild_id, user_id)] = name
    
    async def get_display_name(self, guild_id: int, user_id: int) -> str:
        """Return physicist name or fallback for unknown users."""
        cache_key = (guild_id, user_id)
        if cache_key in self._display_name_cache:
            return self._display_name_cache[cache_key]
        
        # Fallback for unknown users
        return f"Unknown_User_{user_id}"
    
    def get_display_name_sync(self, guild_id: int, user_id: int) -> str:
        """Synchronous version for testing convenience."""
        if user_id in self._physicists:
            return self._physicists[user_id]
        return f"Unknown_User_{user_id}"
    
    async def replace_user_mentions_with_names(self, text: str, guild_id: int) -> str:
        """Replace user mentions like <@123456> with physicist names."""
        import re
        
        def replace_mention(match):
            user_id = int(match.group(1))
            if user_id in self._physicists:
                return self._physicists[user_id]
            return f"Unknown_User_{user_id}"
        
        # Replace Discord mentions <@123456> with names
        return re.sub(r'<@(\d+)>', replace_mention, text)
    
    def resolve_user_id(self, user_mention: str) -> int | None:
        """Resolve physicist name or mention to user ID."""
        # Handle Discord mentions
        if user_mention.startswith('<@') and user_mention.endswith('>'):
            try:
                return int(user_mention[2:-1])
            except ValueError:
                return None
        
        # Handle name lookup
        for user_id, name in self._physicists.items():
            if name.lower() == user_mention.lower():
                return user_id
        
        return None
    
    @property
    def physicist_ids(self) -> dict[str, int]:
        """Get mapping of physicist names to IDs for test convenience."""
        return {name: user_id for user_id, name in self._physicists.items()}
    
    @property
    def physics_guild_id(self) -> int:
        """Standard physics guild ID for tests."""
        return 19001930
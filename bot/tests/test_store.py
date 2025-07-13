"""
Test double for Store pre-populated with physics chat history.
This provides realistic conversation data for testing memory systems.
"""

import re
from datetime import datetime, timezone, date
from typing import List
from store import Store, ChatMessage
from test_user_resolver import TestUserResolver


class TestStore(Store):
    """Test double Store pre-populated with physics revolution chat history."""
    
    def __init__(self):
        # Don't call super().__init__() since we don't need real database
        self.user_resolver = TestUserResolver()
        self.physics_guild_id = self.user_resolver.physics_guild_id
        self.physicist_ids = self.user_resolver.physicist_ids
        
        # Parse and store the physics chat history
        self._messages = self._parse_physics_chat_history()
        
        # Store for user facts (empty by default for testing)
        self._user_facts = {}
        
        # Store for daily summaries: (guild_id, for_date) -> dict[user_id, summary]
        self._daily_summaries = {}
    
    def _parse_physics_chat_history(self) -> List[ChatMessage]:
        """Parse the physics chat history into ChatMessage objects."""
        messages = []
        
        # Physics chat data with timestamps
        chat_data = [
            # Monday, March 3rd, 1905
            ("1905-03-03 09:15", "Einstein", "Good morning colleagues. I've been pondering the photoelectric effect. Light seems to behave as discrete packets of energy, not continuous waves as we've long assumed."),
            ("1905-03-03 09:18", "Planck", "Albert, your quantum hypothesis is intriguing, but surely you don't mean to abandon wave theory entirely? My work on blackbody radiation suggests energy comes in discrete quanta, but only for oscillators."),
            ("1905-03-03 09:22", "Bohr", "Gentlemen, I believe we're onto something revolutionary here. The atom itself might have quantized energy levels. This could explain hydrogen's spectral lines perfectly."),
            ("1905-03-03 09:25", "Thomson", "Niels, your atomic model is fascinating, but how do you reconcile it with my discovery of the electron? Surely the atom is more like a plum pudding?"),
            ("1905-03-03 09:28", "Rutherford", "J.J., I'm afraid your model doesn't hold up. My gold foil experiments show the atom is mostly empty space with a dense nucleus. The electrons orbit this nucleus."),
            ("1905-03-03 09:31", "Bohr", "Ernest, yes! But classical physics says orbiting electrons should radiate energy and spiral into the nucleus. Quantum mechanics saves us here."),
            ("1905-03-03 09:35", "Schrödinger", "This quantum jumping troubles me deeply. Where is the electron between quantum states? Surely there must be a wave description that's more... continuous?"),
            ("1905-03-03 09:38", "Heisenberg", "Erwin, you're thinking too classically. We must abandon the notion of definite particle trajectories. My uncertainty principle shows position and momentum cannot both be precisely known."),
            ("1905-03-03 09:42", "Born", "Werner is correct. We must interpret the wave function probabilistically. Schrödinger's equation gives us probability amplitudes, not definite positions."),
            ("1905-03-03 09:45", "Einstein", "Max, Werner, I remain uncomfortable with this probabilistic interpretation. \"God does not play dice with the universe.\" There must be hidden variables we haven't discovered."),
            
            # Tuesday, March 4th, 1905
            ("1905-03-04 10:20", "Curie", "Gentlemen, while you debate the quantum nature of light, I've been studying radioactive decay. The energy released suggests mass itself can be converted to energy."),
            ("1905-03-04 10:23", "Einstein", "Marie, precisely! My special relativity shows E=mc². Mass and energy are equivalent. This could explain the enormous energy in radioactive processes."),
            ("1905-03-04 10:26", "Lorentz", "Albert, your time dilation and length contraction seem to follow from the constant speed of light, but the physical interpretation still puzzles me."),
            ("1905-03-04 10:29", "Minkowski", "Hendrik, think of it geometrically. Space and time are unified in a four-dimensional spacetime. What we perceive as \"now\" is just a slice through this manifold."),
            ("1905-03-04 10:32", "Planck", "This relativity business is quite radical. Are you suggesting Newton's absolute space and time are illusions?"),
            ("1905-03-04 10:35", "Einstein", "Max, exactly. There is no absolute reference frame. Physics must be the same for all observers in uniform motion."),
            ("1905-03-04 10:38", "Rutherford", "All this mathematical abstraction is fine, but I trust my experimental results. The nucleus exists, and it's incredibly dense."),
            ("1905-03-04 10:41", "Bohr", "Ernest, your experiments are crucial, but theory must guide us too. My model explains why atoms are stable and why they emit specific wavelengths."),
            
            # Wednesday, March 5th, 1905
            ("1905-03-05 14:15", "Heisenberg", "I've been developing a new matrix mechanics. We must work only with observable quantities - energy levels, transition probabilities."),
            ("1905-03-05 14:18", "Schrödinger", "Werner, your matrices are terribly abstract. My wave equation is more intuitive - the electron is a standing wave around the nucleus."),
            ("1905-03-05 14:21", "Dirac", "Gentlemen, I believe I can unify your approaches. My equation combines special relativity with quantum mechanics and predicts antimatter."),
            ("1905-03-05 14:24", "Pauli", "Paul, your equation is elegant, but I'm concerned about the implications. Antimatter? Also, my exclusion principle explains the periodic table structure."),
            ("1905-03-05 14:27", "Einstein", "This quantum mechanics, as it stands, seems incomplete. There must be a deeper theory that restores determinism."),
            ("1905-03-05 14:30", "Bohr", "Albert, we must accept that nature is fundamentally probabilistic at the quantum level. Complementarity shows us that wave and particle descriptions are both necessary."),
            ("1905-03-05 14:33", "Born", "The Copenhagen interpretation is our best framework. The wave function collapse upon measurement is a fundamental feature of reality."),
            ("1905-03-05 14:36", "Schrödinger", "Max, this collapse troubles me. If a quantum system can be in superposition, what about macroscopic objects? Consider a cat in a box..."),
            
            # Thursday, March 6th, 1905
            ("1905-03-06 11:45", "Curie", "My studies of radium show that radioactive decay follows statistical laws, not deterministic ones. Individual atoms decay randomly."),
            ("1905-03-06 11:48", "Rutherford", "Marie, I've been bombarding nitrogen with alpha particles. I can actually transmute elements! Alchemy was wrong in method but right in principle."),
            ("1905-03-06 11:51", "Thomson", "Ernest, your nuclear transmutation is remarkable. It suggests the atom has substructure we're only beginning to understand."),
            ("1905-03-06 11:54", "Planck", "Colleagues, I introduced the quantum reluctantly to solve the ultraviolet catastrophe. Now it seems to govern all of microscopic physics."),
            ("1905-03-06 11:57", "Einstein", "Max, your constant h is appearing everywhere. In photoelectric effect, in Bohr's atom, in de Broglie's matter waves."),
            ("1905-03-06 12:00", "de_Broglie", "Albert, if light has particle properties, perhaps matter has wave properties. My wavelength formula λ = h/p applies to all matter."),
            ("1905-03-06 12:03", "Heisenberg", "Louis, brilliant! This wave-particle duality is fundamental. My uncertainty principle follows naturally from wave packets."),
            ("1905-03-06 12:06", "Bohr", "The correspondence principle helps bridge classical and quantum physics. For large quantum numbers, quantum predictions approach classical ones."),
            
            # Friday, March 7th, 1905
            ("1905-03-07 16:20", "Einstein", "I've been working on extending relativity to include gravity. I believe gravity is the curvature of spacetime itself."),
            ("1905-03-07 16:23", "Minkowski", "Albert, your general relativity is geometrically beautiful. Mass-energy tells spacetime how to curve, and curved spacetime tells matter how to move."),
            ("1905-03-07 16:26", "Lorentz", "This is quite beyond my ether theory. You're saying gravity isn't a force but geometry?"),
            ("1905-03-07 16:29", "Planck", "Albert, if gravity affects spacetime, what happens to energy conservation in an expanding universe?"),
            ("1905-03-07 16:32", "Einstein", "Max, energy conservation becomes local rather than global. The universe itself might be dynamic, not static as we assumed."),
            ("1905-03-07 16:35", "Rutherford", "While you theorists debate cosmology, I'm splitting atoms. The forces holding the nucleus together are incredibly strong."),
            ("1905-03-07 16:38", "Bohr", "Ernest, quantum mechanics might explain nuclear forces too. Perhaps there are new fundamental interactions beyond electromagnetic."),
            
            # Saturday, March 8th, 1905
            ("1905-03-08 09:00", "Schrödinger", "I remain troubled by the measurement problem. When exactly does the wave function collapse? What constitutes a measurement?"),
            ("1905-03-08 09:03", "Heisenberg", "Erwin, the observer cannot be separated from the system. The act of measurement disturbs the quantum state necessarily."),
            ("1905-03-08 09:06", "Einstein", "This observer-dependence seems to make physics subjective. Surely reality exists independently of our observations?"),
            ("1905-03-08 09:09", "Bohr", "Albert, at the quantum level, the classical subject-object distinction breaks down. We must accept complementarity."),
            ("1905-03-08 09:12", "Pauli", "The spin-statistics connection is puzzling. Fermions obey exclusion, bosons don't. This seems to govern all matter organization."),
            ("1905-03-08 09:15", "Dirac", "Wolfgang, my equation naturally predicts spin-1/2 particles. The mathematics seems to know more than we do."),
            ("1905-03-08 09:18", "Born", "Paul, mathematics is our guide to physical truth. The formalism often reveals aspects of reality we hadn't anticipated."),
            ("1905-03-08 09:21", "Curie", "Gentlemen, while you debate interpretation, radioactivity continues to reveal new elements and phenomena. Experiment must anchor theory."),
            
            # Sunday, March 9th, 1905
            ("1905-03-09 20:30", "Einstein", "Looking back on this week's discussions, we've witnessed the birth of a new physics. Classical certainties have given way to quantum probabilities."),
            ("1905-03-09 20:33", "Planck", "Albert, we've reluctantly dismantled the edifice of 19th-century physics. What we're building in its place is both more powerful and more mysterious."),
            ("1905-03-09 20:36", "Bohr", "The atomic world operates by rules completely foreign to our everyday experience. Complementarity, uncertainty, superposition - these are not failures of understanding but features of reality."),
            ("1905-03-09 20:39", "Heisenberg", "We've learned that asking certain questions - like the exact position and momentum of a particle - is meaningless. Nature herself draws the boundaries of knowledge."),
            ("1905-03-09 20:42", "Schrödinger", "Though I disagree with some interpretations, I cannot deny the theory's predictive power. My equation describes atoms with unprecedented accuracy."),
            ("1905-03-09 20:45", "Rutherford", "Your theories explain my experiments beautifully. The quantum nucleus, governed by probability, yet yielding such precise statistical predictions."),
            ("1905-03-09 20:48", "Curie", "We've discovered that matter and energy are interconvertible, that space and time are unified, that atoms are mostly empty space with dense nuclei."),
            ("1905-03-09 20:51", "Dirac", "The marriage of special relativity and quantum mechanics has revealed antimatter and predicted the existence of particles we haven't yet observed."),
            ("1905-03-09 20:54", "Born", "Most remarkably, we've learned that the universe is fundamentally probabilistic, not deterministic. This challenges our deepest philosophical assumptions."),
            ("1905-03-09 20:57", "Einstein", "Though I remain skeptical of the probabilistic interpretation, I cannot deny the theory's success. Perhaps future generations will find the deeper deterministic theory I believe must exist."),
            ("1905-03-09 21:00", "Bohr", "Albert, perhaps the lesson is that nature is under no obligation to conform to our classical intuitions. The quantum world is strange, but it is our world."),
        ]
        
        # Convert to ChatMessage objects
        message_id_counter = 10000
        channel_id = 1905  # Physics revolution year
        
        for timestamp_str, physicist_name, content in chat_data:
            if physicist_name not in self.physicist_ids:
                continue  # Skip unknown physicists
                
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            user_id = self.physicist_ids[physicist_name]
            
            # Process mentions in content (replace physicist names with Discord mentions)
            processed_content = self._process_mentions(content)
            
            message = ChatMessage(
                guild_id=self.physics_guild_id,
                channel_id=channel_id,
                message_id=message_id_counter,
                user_id=user_id,
                message_text=processed_content,
                timestamp=timestamp
            )
            messages.append(message)
            message_id_counter += 1
        
        return messages
    
    def _process_mentions(self, content: str) -> str:
        """Convert physicist names in content to Discord mentions."""
        # Map common name patterns to physicist names
        name_patterns = {
            "Albert": "Einstein",
            "Max": "Planck", 
            "Niels": "Bohr",
            "J.J.": "Thomson",
            "Ernest": "Rutherford",
            "Erwin": "Schrödinger",
            "Werner": "Heisenberg",
            "Marie": "Curie",
            "Hendrik": "Lorentz",
            "Paul": "Dirac",  # Note: Could also be Pauli, but Dirac is more common in context
            "Wolfgang": "Pauli",
            "Louis": "de_Broglie",
        }
        
        processed_content = content
        for name_pattern, physicist_name in name_patterns.items():
            if physicist_name in self.physicist_ids:
                user_id = self.physicist_ids[physicist_name]
                # Replace name with Discord mention
                processed_content = re.sub(
                    rf'\b{re.escape(name_pattern)}\b',
                    f'<@{user_id}>',
                    processed_content
                )
        
        return processed_content
    
    async def get_chat_messages_for_date(self, guild_id: int, for_date: date) -> List[ChatMessage]:
        """Get chat messages for a specific date."""
        if guild_id != self.physics_guild_id:
            return []
        
        # Filter messages by date
        filtered_messages = []
        for message in self._messages:
            if message.timestamp.date() == for_date:
                filtered_messages.append(message)
        
        return filtered_messages
    
    async def get_user_facts(self, guild_id: int, user_id: int) -> str | None:
        """Get stored facts for a user."""
        if guild_id != self.physics_guild_id:
            return None
        return self._user_facts.get(user_id)
    
    async def add_chat_message(self, guild_id: int, channel_id: int, message_id: int, 
                             user_id: int, message_text: str, timestamp: datetime) -> None:
        """Add a chat message to the store."""
        message = ChatMessage(
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=message_id,
            user_id=user_id,
            message_text=message_text,
            timestamp=timestamp
        )
        self._messages.append(message)
    
    def set_user_facts(self, guild_id: int, user_id: int, facts: str) -> None:
        """Set facts for a user (test helper method)."""
        if guild_id == self.physics_guild_id:
            self._user_facts[user_id] = facts
    
    async def save_user_facts(self, guild_id: int, user_id: int, facts: str) -> None:
        """Save facts for a user (overrides parent to avoid telemetry dependency)."""
        self.set_user_facts(guild_id, user_id, facts)
    
    def get_message_count_for_date(self, for_date: date) -> int:
        """Get count of messages for a specific date (test helper)."""
        count = 0
        for message in self._messages:
            if message.timestamp.date() == for_date:
                count += 1
        return count
    
    def get_active_physicists_for_date(self, for_date: date) -> List[str]:
        """Get list of physicist names active on a specific date (test helper)."""
        active_user_ids = set()
        for message in self._messages:
            if message.timestamp.date() == for_date:
                active_user_ids.add(message.user_id)
        
        # Convert back to names
        physicist_names = []
        for user_id in active_user_ids:
            for name, pid in self.physicist_ids.items():
                if pid == user_id:
                    physicist_names.append(name)
                    break
        
        return sorted(physicist_names)
    
    @property
    def total_message_count(self) -> int:
        """Total number of messages in the physics chat history."""
        return len(self._messages)
    
    @property
    def date_range(self) -> tuple[date, date]:
        """Get the date range of the physics chat history."""
        if not self._messages:
            return None, None
        
        dates = [msg.timestamp.date() for msg in self._messages]
        return min(dates), max(dates)
    
    async def has_chat_messages_for_date(self, guild_id: int, for_date: date) -> bool:
        """Check if any chat messages exist for a specific guild and date (test implementation)."""
        messages = await self.get_chat_messages_for_date(guild_id, for_date)
        return len(messages) > 0
    
    async def get_daily_summaries(self, guild_id: int, for_date: date) -> dict[int, str]:
        """Get daily summaries for all users on a specific date (test implementation)."""
        cache_key = (guild_id, for_date)
        return self._daily_summaries.get(cache_key, {})
    
    async def save_daily_summaries(self, guild_id: int, for_date: date, summaries_dict: dict[int, str]) -> None:
        """Save daily summaries for multiple users on a specific date (test implementation)."""
        cache_key = (guild_id, for_date)
        self._daily_summaries[cache_key] = summaries_dict
CREATE TABLE messages (
    message_id BIGINT PRIMARY KEY,
    language_code TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE jokes (
    source_message_id BIGINT REFERENCES messages(message_id) ON DELETE CASCADE,
    joke_message_id BIGINT REFERENCES messages(message_id) ON DELETE CASCADE,
    reaction_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_message_id, joke_message_id)
);

CREATE TABLE guild_configs (
    guild_id BIGINT PRIMARY KEY,
    archive_channel_id BIGINT DEFAULT 0,
    delete_jokes_after_minutes INTEGER DEFAULT 0,
    downvote_reaction_threshold INTEGER DEFAULT 0,
    enable_country_jokes BOOLEAN DEFAULT TRUE
);

CREATE TABLE user_facts (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    memory_blob TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE chat_history (
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    message_text TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    reply_to_id BIGINT NULL
);

CREATE INDEX idx_chat_history_guild_timestamp ON chat_history (guild_id, timestamp);

CREATE TABLE daily_chat_summaries (
    guild_id BIGINT NOT NULL,
    for_date DATE NOT NULL,
    user_id BIGINT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, for_date, user_id)
);

-- Insert messages
INSERT INTO messages VALUES
(1, 'en', 'My son is bioterrorising me. He walked in my office, sat down for a few minutes, farted, and then immediately left'),
(2, 'en', 'Ur mom is bioterrorising me. She walked in my office, sat down for a few minutes, farted, and then immediately left'),
(3, 'en', 'So Trump outlawed LGBT and there''s no protests whatsoever, it''s as if it was never a big movement and in reality people don''t support it unless forced to'),
(4, 'en', 'So Trump outlawed ur mom and there''s no protests whatsoever, it''s as if it was never a big movement and in reality people don''t support her unless forced to'),
(5, 'en', 'Jim likes cake.'),
(6, 'en', 'Jim likes ur mom.'),
(7, 'en', 'tldr, its basically switch 1 with some extra coloured plastic'),
(8, 'en', 'tldr, its basically ur mom with some extra coloured plastic'),
(9, 'en', 'Phones used to be 500$ now it''s 1500$'),
(10, 'en', 'Ur mom used to be 500$ now it''s 1500$'),
(11, 'en', 'Everyone knows that a tablet that can''t do FaceTime is dead on arrival'),
(12, 'en', 'Everyone knows that ur mom can do FaceTime'),
(13, 'en', 'Ouch. You been putting strange objects in your ear?'),
(14, 'en', 'ur mom has been putting strange objects in her ear'),
(15, 'en', 'Apparently it''s a trophy for getting all achievements'),
(16, 'en', 'It''s a trophy for getting ur mom'),
(17, 'en', 'He''s harder in multiplayer'),
(18, 'en', 'Ur mom''s harder in multiplayer'),
(21, 'en', 'just wait till you see what''s on the roadmap'),
(22, 'en', 'Ur mom''s on the roadmap'),
(23, 'en', 'i doubt that would fit on 4TB'),
(24, 'en', 'i doubt ur mom would fit on 4TB'),
(25, 'en', 'is it possible you aren''t connected to their network somehow? like with a firewall blocking it'),
(26, 'en', 'Ur mom''s blocking it'),
(27, 'en', 'I use USB-C to DP cable'),
(28, 'en', 'I use USB-C to DP ur mom'),
(29, 'en', 'yeah.. and I think our standards get lower than they used to be, thx to netflix series'),
(30, 'en', 'Ur mom''s standards get lower than they used to be'),
(31, 'en', 'How was it down there?'),
(32, 'en', 'How was ur mom down there'),
(33, 'en', 'Nissin has good cheese cake, it''s in the freezer section'),
(34, 'en', 'ur mom''s in the freezer section'),
(35, 'en', 'fun story: As I was taking the garbage downstairs last night (2am), there was a huge (female I guess?) mosquito in the elevator. It was flying around, I really didn''t want it to land on me, so I whacked it with the garbage bag, it died. I walk out of the elevator (and I make note of where it died), drop off the garbage bag (this takes 20 seconds), and go back in the elevator.....I check the mosquito......it''s gone, no where to be seen........must be a big spider/something living in our elevator.'),
(36, 'en', 'ur mom got whacked at 2am last night'),
(37, 'en', 'Musk is working on it'),
(38, 'en', 'Musk is working on ur mom'),
(39, 'en', 'sound via bone conduction'),
(40, 'en', 'I typically use ur mom for sound via bone conduction'),
(41, 'en', 'this website is too much fun'),
(42, 'en', 'ur mom is too much fun'),
(43, 'en', 'Yes, everything can be done with a couple of command lines'),
(44, 'en', 'ur mom can be done with a couple of command lines'),
(45, 'en', 'that is a predicament. Maybe your charting library supports doing it somehow though?'),
(46, 'en', 'that is a predicament. Maybe your charting library supports doing ur mom somehow though?'),
(47, 'en', 'you just reminded me I need to clean ours. hasn''t been cleaned in like 8 months'),
(48, 'en', 'ur mom hasn''t been cleaned in like 8 months'),
(49, 'en', 'I was just reading, they think it''s falling because US interest rates are going up and Japan''s are ultra-low'),
(50, 'en', 'ur mom''s interest rate is going down'),
(51, 'en', 'what is that blue ball at the base of it?'),
(52, 'en', 'prolly spent the night with ur mom'),
(53, 'en', 'I''m seeing Top Gun on Sunday!'),
(54, 'en', 'Ur mom''s seeing top gun on Sunday, whatever that might mean'),
(55, 'en', 'is there anything you own that you would NOT recommend to anyone ?'),
(56, 'en', 'yes, ur mom'),
(57, 'en', 'Ooo, so you need the sausages asap'),
(58, 'en', 'ur mom need the sausages asap'),
(59, 'en', 'yes , I ended up using the biggest size'),
(60, 'en', 'ur mom ended up using the biggest size'),
(61, 'en', 'that would be some weak ass theory'),
(62, 'en', 'ur mom''s a weak ass theory'),
(63, 'en', 'whats leaking?'),
(64, 'en', 'ur mom'),
(65, 'en', 'And I tried touching midnight color at a store - my hands were just leaving wet marks on it'),
(66, 'en', 'ur mom leaves wet marks'),
(67, 'en', 'Markets are having a good rally recently'),
(68, 'en', 'ur mom''s having a good rally'),
(69, 'en', 'Looks like a Boston dynamics robot'),
(70, 'en', 'ur mom looks like a Boston dynamics robot'),
(71, 'en', 'best show I''ve seen in a while'),
(72, 'en', 'ur mom''s the best show I''ve seen in a while'),
(73, 'en', 'GCC isn''t exactly a day job'),
(74, 'en', 'ur mom isn''t exactly a day job'),
(75, 'en', 'do you guys have the ability to pin messages?'),
(76, 'en', 'I have the ability to pin ur mom');

-- Insert joke relationships
INSERT INTO jokes (source_message_id, joke_message_id, reaction_count) VALUES
(1, 2, 1),
(3, 4, 1),
(5, 6, 1),
(7, 8, 1),
(9, 10, 1),
(11, 12, 1),
(13, 14, 1),
(15, 16, 1),
(17, 18, 1),
(21, 22, 1),
(23, 24, 1),
(25, 26, 1),
(27, 28, 1),
(29, 30, 1),
(31, 32, 1),
(33, 34, 1),
(35, 36, 1),
(37, 38, 1),
(39, 40, 1),
(41, 42, 1),
(43, 44, 1),
(45, 46, 1),
(47, 48, 1),
(49, 50, 1),
(51, 52, 1),
(53, 54, 1),
(55, 56, 1),
(57, 58, 1),
(59, 60, 1),
(61, 62, 1),
(63, 64, 1),
(65, 66, 1),
(67, 68, 1),
(69, 70, 1),
(71, 72, 1),
(73, 74, 1),
(75, 76, 1);
DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id INT PRIMARY KEY,
    text TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    price INT NULL
);

INSERT INTO documents (id, text, category, price) VALUES
    (1, 'Example product with three rooms, breakfast, Wi-Fi, and a swimming pool. Prices start at 120 USD per night.', 'lodging', 120),
    (2, 'Example guided service lasting one full day. It starts at 05:00 and includes private transportation and a professional guide.', 'experience', NULL),
    (3, 'Example FAQ: Customers can review opening hours, accessibility information, available payment methods, and cancellation terms before booking.', 'support', NULL);

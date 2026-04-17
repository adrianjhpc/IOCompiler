#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

const int WIDTH = 4000;
const int HEIGHT = 4000;
const int TOTAL_PIXELS = WIDTH * HEIGHT;
const char* FILENAME = "output.ppm";
const char* EXPECTED_HEADER = "P6\n4000 4000\n255\n";

// -----------------------------------------------------------------------------
// Correctness verification
// -----------------------------------------------------------------------------
#include <cstdint>
// ... keep your other includes ...

bool verify_output() {
    std::cout << "Verifying binary correctness...\n";

    std::ifstream file(FILENAME, std::ios::binary);
    if (!file) {
        std::cerr << "[FAIL] Could not open output file for verification.\n";
        return false;
    }

    // Verify total file size first (fast sanity check)
    const std::size_t header_len = std::strlen(EXPECTED_HEADER);
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(header_len) +
        static_cast<std::uint64_t>(TOTAL_PIXELS) * 3ULL;

    file.seekg(0, std::ios::end);
    std::uint64_t actual_size = static_cast<std::uint64_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (actual_size != expected_size) {
        std::cerr << "[FAIL] File size mismatch. Expected " << expected_size
                  << " bytes, got " << actual_size << " bytes.\n";
        return false;
    }

    // Verify header
    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), static_cast<std::streamsize>(header_len));
    if (file.gcount() != static_cast<std::streamsize>(header_len)) {
        std::cerr << "[FAIL] Truncated file while reading header.\n";
        return false;
    }
    if (std::memcmp(header_buf.data(), EXPECTED_HEADER, header_len) != 0) {
        std::cerr << "[FAIL] Header corruption detected.\n";
        return false;
    }

    // Verify pixel data (read one row at a time)
    std::vector<std::uint8_t> row_buffer(static_cast<std::size_t>(WIDTH) * 3);
    int pixel_index = 0;

    for (int y = 0; y < HEIGHT; y++) {
        file.read(reinterpret_cast<char*>(row_buffer.data()),
                  static_cast<std::streamsize>(row_buffer.size()));

        if (file.gcount() != static_cast<std::streamsize>(row_buffer.size())) {
            std::cerr << "[FAIL] File truncated early at row " << y << ".\n";
            return false;
        }

        for (int x = 0; x < WIDTH; x++) {
            const std::uint8_t expected_r = static_cast<std::uint8_t>(pixel_index % 255);
            const std::uint8_t expected_g = static_cast<std::uint8_t>((pixel_index / WIDTH) % 255);
            const std::uint8_t expected_b = 128;

            const std::size_t idx = static_cast<std::size_t>(x) * 3;
            const std::uint8_t got_r = row_buffer[idx + 0];
            const std::uint8_t got_g = row_buffer[idx + 1];
            const std::uint8_t got_b = row_buffer[idx + 2];

            if (got_r != expected_r || got_g != expected_g || got_b != expected_b) {
                std::cerr << "[FAIL] Pixel corruption at index " << pixel_index
                          << " (X: " << x << ", Y: " << y << ").\n";
                std::cerr << "Expected: [" << (unsigned)expected_r << ", "
                          << (unsigned)expected_g << ", " << (unsigned)expected_b << "]\n";
                std::cerr << "Got:      [" << (unsigned)got_r << ", "
                          << (unsigned)got_g << ", " << (unsigned)got_b << "]\n";
                return false;
            }

            pixel_index++;
        }
    }

    std::cout << "[PASS] 100% Data Integrity! (" << pixel_index << " pixels verified).\n";
    return true;
}


// -----------------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------------
int main() {
    int fd = open(FILENAME, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    write(fd, EXPECTED_HEADER, strlen(EXPECTED_HEADER));

    std::cout << "Generating 4000x4000 image...\n";
    
    auto start = std::chrono::high_resolution_clock::now();

    uint8_t pixel[3];
    
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        pixel[0] = (i % 255);           // Red
        pixel[1] = ((i / WIDTH) % 255); // Green
        pixel[2] = 128;                 // Blue

        // Unoptimized: 3 system calls. Optimized: 1 system call.
        write(fd, pixel, 1);
        write(fd, pixel + 1, 1);
        write(fd, pixel + 2, 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " seconds\n";

    // Flush and close the file to disk so we can safely verify it
    close(fd);

    // Run the correctness checker
    if (!verify_output()) {
        return 1; // Exit with error if our compiler pass corrupted the logic
    }

    return 0;
}

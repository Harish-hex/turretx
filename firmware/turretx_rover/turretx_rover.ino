/*
  TURRETX rover: BLE WASD control (unchanged) + data over WiFi
  ============================================================
  - Driving: BLE from the browser controller, exactly as before (W/A/S/D/X, 500 ms safety stop).
  - Data: over WiFi UDP on the hotspot (BLE no longer carries telemetry):
      1. IFF beacon: the laptop broadcasts a challenge, the rover answers with a SIGNED claim of
         where it is and how it moves. The laptop verifies it and matches it to a camera track.
      2. Telemetry: encoder + pose, signed, 4x per second.
  - Position = dead reckoning: AS5600 wheel distance + MPU6050 gyro heading, from a START POSE.
    During A/D pivot turns the wheels spin opposite ways but the rover doesn't travel, so only the
    heading changes then (the encoder is ignored for position while pivoting).

  WiFi UDP ports (laptop on the same hotspot):
    4210  laptop -> broadcast   challenge  {"n":"<nonce hex>","t":<laptop ms>}
    4211  rover  -> laptop      IFF reply  {"payload":"<hex>","sig":"<hex>"}
          payload = {"id":"F-1","n":"<nonce>","t":<challenge ms>,"x":<mm>,"y":<mm>,"vx":<mm/s>,"vy":<mm/s>,"rssi":<dBm x10>}
          (integers only, must match turretx/wire.py; "t" echoes the laptop's time)
    4212  rover  -> laptop      telemetry  {"payload":"<hex>","sig":"<hex>"}
          payload = {"id":"F-1","type":"tel","seq":..,"cmd":"W","raw":..,"rev_milli":..,"mag":1,
                     "x":<mm>,"y":<mm>,"h":<deg x10>,"v":<mm/s>}
          (sent to the laptop once a challenge has been heard; broadcast until then)
  sig = HMAC-SHA256(key, payload bytes exactly as sent), lowercase hex.

  Arena frame (same as the laptop): origin = TOP-LEFT corner of the clicked/marked area,
  x -> top-right, y -> bottom-left, cm. Heading 0 = facing +x, 90 deg = facing +y (y points DOWN
  in the camera image).

  Arduino IDE: board "ESP32S3 Dev Module", Partition Scheme "Huge APP (3MB No OTA)" (BLE + WiFi
  don't fit the default partition). No extra libraries.
*/

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "mbedtls/md.h"

// ==================================================
// TURRETX DATA LINK CONFIG (edit these)
// ==================================================

const char* WIFI_SSID     = "Galaxy S23 D779";   // the hotspot the laptop is on
const char* WIFI_PASSWORD = "CHE";
const char* RF_ID         = "F-1";              // must match the friendly id on the laptop
const char* RF_KEY        = "DEMO-KEY-turretx"; // = laptop rf.key / TURRETX_RF_KEY (demo key only)

const uint16_t PORT_CHALLENGE = 4210;
const uint16_t PORT_REPLY     = 4211;
const uint16_t PORT_TELEMETRY = 4212;

// Start pose: put the rover HERE facing THIS way, then power on (or press BOOT to re-zero).
const float START_X_CM        = 20.0;
const float START_Y_CM        = 20.0;
const float START_HEADING_DEG = 0.0;            // 0 = facing the +x direction (towards top-right)

const float WHEEL_DIAMETER_CM = 6.5;            // MEASURE the wheel the AS5600 magnet is on
const float ENCODER_SIGN      = 1.0;            // flip to -1 if driving W makes the distance go DOWN
// Turn the rover CLOCKWISE as seen from the camera: the logged heading must INCREASE.
// If it decreases, flip GYRO_SIGN to +1.
const float GYRO_SIGN         = -1.0;
const int   BOOT_BUTTON       = 0;
const float STATIONARY_CM_S   = 2.0;            // below this the rover reports zero velocity

// ==================================================
// L298N MOTOR DRIVER PINS
// ==================================================

#define IN1 4
#define IN2 5
#define IN3 6
#define IN4 7

// ==================================================
// MPU6050 - PRIMARY I2C BUS
// ==================================================

#define MPU_SDA_PIN 17
#define MPU_SCL_PIN 18
#define MPU_ADDR 0x68

// ==================================================
// AS5600 - SECONDARY I2C BUS
// ==================================================

#define ENC_SDA_PIN 41
#define ENC_SCL_PIN 42
#define AS5600_ADDR 0x36

TwoWire EncoderWire(1);

// AS5600 registers
#define AS5600_STATUS_REG 0x0B
#define AS5600_ANGLE_REG  0x0C

// ==================================================
// BLUETOOTH UUIDs (control only)
// ==================================================

#define SERVICE_UUID \
  "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

#define COMMAND_UUID \
  "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

#define STATUS_UUID \
  "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

BLECharacteristic *statusCharacteristic = nullptr;

bool deviceConnected = false;

// ==================================================
// TIMERS AND COMMANDS
// ==================================================

char currentCommand = 'X';

unsigned long lastCommandAt = 0;
unsigned long lastMPUReadAt = 0;
unsigned long lastEncoderReadAt = 0;
unsigned long lastTelemetryAt = 0;

const unsigned long COMMAND_TIMEOUT_MS = 500;
const unsigned long MPU_INTERVAL_MS = 200;
const unsigned long ENCODER_INTERVAL_MS = 50;
const unsigned long TELEMETRY_INTERVAL_MS = 250;

// ==================================================
// ENCODER VARIABLES
// ==================================================

int previousRawAngle = -1;
long encoderCounts = 0;

int rawAngle = -1;
float angleDegrees = 0.0;
float totalAngleDegrees = 0.0;
float revolutions = 0.0;

bool encoderDetected = false;
bool magnetDetected = false;

// ==================================================
// DEAD RECKONING + WIFI STATE
// ==================================================

float poseX = 0, poseY = 0, headingRad = 0;     // arena cm, radians
float speedCmS = 0;                             // + forward, - backward
float gyroBiasDps = 0;
float lastDistCm = 0, distAtLastSpeed = 0;
unsigned long lastGyroUs = 0, lastSpeedAt = 0, stillSinceMs = 0;

WiFiUDP udpIn, udpOut;
bool netStarted = false;
bool laptopKnown = false;
IPAddress laptopIP;
uint32_t telemetrySeq = 0;

// ==================================================
// MOTOR CONTROL
// ==================================================

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void moveForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void turnLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void turnRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

// ==================================================
// BLE COMMAND HANDLER
// ==================================================

void setCommand(char command) {
  if (command >= 'a' && command <= 'z') {
    command = command - 'a' + 'A';
  }

  switch (command) {
    case 'W':
      currentCommand = 'W';
      moveForward();
      break;

    case 'S':
      currentCommand = 'S';
      moveBackward();
      break;

    case 'A':
      currentCommand = 'A';
      turnLeft();
      break;

    case 'D':
      currentCommand = 'D';
      turnRight();
      break;

    case 'X':
    default:
      currentCommand = 'X';
      stopMotors();
      break;
  }

  lastCommandAt = millis();

  Serial.print("Command: ");
  Serial.println(currentCommand);
}

// ==================================================
// BLE CALLBACKS
// ==================================================

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    deviceConnected = true;
    Serial.println("Bluetooth connected");
  }

  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    currentCommand = 'X';
    stopMotors();

    Serial.println("Bluetooth disconnected - motors stopped");

    BLEDevice::startAdvertising();
  }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    String value = characteristic->getValue();

    if (value.length() > 0) {
      setCommand(value.charAt(0));
    }
  }
};

// ==================================================
// GENERIC I2C READ HELPER
// ==================================================

bool readI2CBytes(
    TwoWire &bus,
    uint8_t address,
    uint8_t reg,
    uint8_t *buffer,
    uint8_t length) {

  bus.beginTransmission(address);
  bus.write(reg);

  if (bus.endTransmission(false) != 0) {
    return false;
  }

  size_t received = bus.requestFrom(address, length);

  if (received != length) {
    while (bus.available()) {
      bus.read();
    }
    return false;
  }

  for (uint8_t i = 0; i < length; i++) {
    buffer[i] = bus.read();
  }

  return true;
}

int16_t combineBytes(uint8_t highByte, uint8_t lowByte) {
  return (int16_t)(
    ((uint16_t)highByte << 8) | lowByte
  );
}

// ==================================================
// MPU6050
// ==================================================

bool initializeMPU6050() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: MPU6050 not detected");
    return false;
  }

  // +-250 dps gyro range and a ~44 Hz low-pass filter for clean heading integration
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x1B); Wire.write(0x00); Wire.endTransmission();
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x1A); Wire.write(0x03); Wire.endTransmission();

  Serial.println("MPU6050 initialized");
  return true;
}

void readMPU6050() {
  uint8_t data[14];

  if (!readI2CBytes(Wire, MPU_ADDR, 0x3B, data, 14)) {
    Serial.println("MPU6050 read error");
    return;
  }

  int16_t axRaw = combineBytes(data[0], data[1]);
  int16_t ayRaw = combineBytes(data[2], data[3]);
  int16_t azRaw = combineBytes(data[4], data[5]);

  int16_t tempRaw = combineBytes(data[6], data[7]);

  int16_t gxRaw = combineBytes(data[8], data[9]);
  int16_t gyRaw = combineBytes(data[10], data[11]);
  int16_t gzRaw = combineBytes(data[12], data[13]);

  float ax = axRaw / 16384.0;
  float ay = ayRaw / 16384.0;
  float az = azRaw / 16384.0;

  float gx = gxRaw / 131.0;
  float gy = gyRaw / 131.0;
  float gz = gzRaw / 131.0;

  float temperature = tempRaw / 340.0 + 36.53;

  Serial.print("MPU | ACC[g] ");
  Serial.print(ax, 2);
  Serial.print(", ");
  Serial.print(ay, 2);
  Serial.print(", ");
  Serial.print(az, 2);

  Serial.print(" | GYRO[dps] ");
  Serial.print(gx, 2);
  Serial.print(", ");
  Serial.print(gy, 2);
  Serial.print(", ");
  Serial.print(gz, 2);

  Serial.print(" | TEMP ");
  Serial.print(temperature, 1);
  Serial.println(" C");
}

// Fast gyro Z read for heading (the full readMPU6050() above is only for the 5 Hz debug print)
bool readGyroZ(float &dps) {
  uint8_t data[2];
  if (!readI2CBytes(Wire, MPU_ADDR, 0x47, data, 2)) {
    return false;
  }
  dps = combineBytes(data[0], data[1]) / 131.0;
  return true;
}

void calibrateGyro() {
  Serial.println("Gyro calibration: keep the rover STILL for 2 s...");
  float sum = 0;
  int n = 0;
  for (int i = 0; i < 400; i++) {
    float g;
    if (readGyroZ(g)) { sum += g; n++; }
    delay(5);
  }
  gyroBiasDps = n ? sum / n : 0;
  Serial.print("Gyro bias: ");
  Serial.print(gyroBiasDps, 3);
  Serial.println(" dps");
}

// ==================================================
// AS5600 INITIALIZATION
// ==================================================

bool initializeAS5600() {
  EncoderWire.beginTransmission(AS5600_ADDR);

  if (EncoderWire.endTransmission() == 0) {
    Serial.println("AS5600 detected at 0x36");
    return true;
  }

  Serial.println("ERROR: AS5600 not detected");
  return false;
}

// ==================================================
// AS5600 ANGLE READING
// ==================================================

int readAS5600Angle() {
  uint8_t data[2];

  if (!readI2CBytes(
        EncoderWire,
        AS5600_ADDR,
        AS5600_ANGLE_REG,
        data,
        2)) {
    return -1;
  }

  return ((data[0] & 0x0F) << 8) | data[1];
}

// ==================================================
// DEAD RECKONING
// ==================================================

float encoderDistanceCm() {
  return ENCODER_SIGN * (encoderCounts / 4096.0) * PI * WHEEL_DIAMETER_CM;
}

void resetPose() {
  poseX = START_X_CM;
  poseY = START_Y_CM;
  headingRad = START_HEADING_DEG * DEG_TO_RAD;
  lastDistCm = encoderDistanceCm();
  distAtLastSpeed = lastDistCm;
  speedCmS = 0;
  Serial.println("Pose reset to the start pose");
}

// Heading from the gyro, ~200 Hz
void updateHeading() {
  unsigned long nowUs = micros();
  float dt = (nowUs - lastGyroUs) * 1e-6;
  if (dt < 0.005) {
    return;
  }
  lastGyroUs = nowUs;

  float g;
  if (!readGyroZ(g)) {
    return;
  }
  float wz = g - gyroBiasDps;
  if (fabs(wz) < 0.5) {
    wz = 0;                                     // deadband: no slow drift while parked
  }
  headingRad += GYRO_SIGN * wz * dt * DEG_TO_RAD;
}

// Position from the encoder, after every encoder read (20 Hz)
void updatePosition() {
  float dist = encoderDistanceCm();
  float dd = dist - lastDistCm;
  lastDistCm = dist;

  bool pivoting = (currentCommand == 'A' || currentCommand == 'D');
  if (!pivoting) {
    poseX += dd * cos(headingRad);
    poseY += dd * sin(headingRad);
  }

  unsigned long ms = millis();
  if (ms - lastSpeedAt >= 100) {
    float s = pivoting ? 0 : (dist - distAtLastSpeed) / ((ms - lastSpeedAt) / 1000.0);
    speedCmS = 0.6 * speedCmS + 0.4 * s;
    distAtLastSpeed = dist;
    lastSpeedAt = ms;

    // re-learn the gyro bias while parked
    if (currentCommand == 'X' && fabs(speedCmS) < STATIONARY_CM_S) {
      if (stillSinceMs == 0) {
        stillSinceMs = ms;
      }
      float g;
      if (ms - stillSinceMs > 1000 && readGyroZ(g)) {
        gyroBiasDps = 0.99 * gyroBiasDps + 0.01 * g;
      }
    } else {
      stillSinceMs = 0;
    }
  }
}

// ==================================================
// UPDATE ENCODER POSITION
// ==================================================

void updateEncoder() {
  int newRawAngle = readAS5600Angle();

  if (newRawAngle < 0) {
    encoderDetected = false;
    Serial.println("AS5600 read error");
    return;
  }

  encoderDetected = true;
  rawAngle = newRawAngle;

  angleDegrees = rawAngle * 360.0 / 4096.0;

  if (previousRawAngle >= 0) {
    int difference = rawAngle - previousRawAngle;

    if (difference > 2048) {
      difference -= 4096;
    } else if (difference < -2048) {
      difference += 4096;
    }

    encoderCounts += difference;
  }

  previousRawAngle = rawAngle;

  totalAngleDegrees = encoderCounts * 360.0 / 4096.0;
  revolutions = encoderCounts / 4096.0;

  uint8_t status = 0;

  magnetDetected =
      readI2CBytes(
        EncoderWire,
        AS5600_ADDR,
        AS5600_STATUS_REG,
        &status,
        1)
      && ((status & 0x20) != 0);

  updatePosition();

  Serial.print("ENCODER | Raw: ");
  Serial.print(rawAngle);

  Serial.print(" | Angle: ");
  Serial.print(angleDegrees, 2);
  Serial.print(" deg");

  Serial.print(" | Total: ");
  Serial.print(totalAngleDegrees, 2);
  Serial.print(" deg");

  Serial.print(" | Revolutions: ");
  Serial.print(revolutions, 3);

  Serial.print(" | Magnet: ");
  Serial.print(magnetDetected ? "Detected" : "Not detected");

  Serial.print(" | POSE x ");
  Serial.print(poseX, 1);
  Serial.print(" y ");
  Serial.print(poseY, 1);
  Serial.print(" hdg ");
  Serial.print(headingRad * RAD_TO_DEG, 0);
  Serial.print(" v ");
  Serial.println(speedCmS, 1);
}

// ==================================================
// WIFI: SIGNED PACKETS
// ==================================================

void hmacHex(const char *msg, size_t len, char out[65]) {
  uint8_t mac[32];
  mbedtls_md_context_t ctx;
  mbedtls_md_init(&ctx);
  mbedtls_md_setup(&ctx, mbedtls_md_info_from_type(MBEDTLS_MD_SHA256), 1);
  mbedtls_md_hmac_starts(&ctx, (const uint8_t *)RF_KEY, strlen(RF_KEY));
  mbedtls_md_hmac_update(&ctx, (const uint8_t *)msg, len);
  mbedtls_md_hmac_finish(&ctx, mac);
  mbedtls_md_free(&ctx);
  for (int i = 0; i < 32; i++) {
    sprintf(out + 2 * i, "%02x", mac[i]);
  }
  out[64] = 0;
}

// Sends {"payload":"<hex>","sig":"<hex>"}; the signature covers the payload bytes exactly
void sendSigned(IPAddress ip, uint16_t port, const char *payload, int len) {
  char sig[65];
  hmacHex(payload, len, sig);
  char hex[2 * 256 + 1];
  for (int i = 0; i < len; i++) {
    sprintf(hex + 2 * i, "%02x", (uint8_t)payload[i]);
  }
  udpOut.beginPacket(ip, port);
  udpOut.printf("{\"payload\":\"%s\",\"sig\":\"%s\"}", hex, sig);
  udpOut.endPacket();
}

// ==================================================
// WIFI: IFF CHALLENGE -> SIGNED CLAIM
// ==================================================

// Minimal parse of {"n":"<hex>","t":<int>}; anything else is ignored
bool parseChallenge(const char *s, char nonce[33], long long &tMs) {
  const char *n = strstr(s, "\"n\":\"");
  const char *t = strstr(s, "\"t\":");
  if (!n || !t) {
    return false;
  }
  n += 5;
  int i = 0;
  while (n[i] && n[i] != '"' && i < 32) {
    if (!isxdigit((unsigned char)n[i])) {
      return false;
    }
    nonce[i] = n[i];
    i++;
  }
  if (i == 0 || n[i] != '"') {
    return false;
  }
  nonce[i] = 0;
  tMs = atoll(t + 4);
  return tMs >= 0;
}

void answerChallenge(const char *nonce, long long tMs, IPAddress laptop) {
  float vx = 0, vy = 0;
  if (fabs(speedCmS) >= STATIONARY_CM_S) {
    vx = speedCmS * cos(headingRad);
    vy = speedCmS * sin(headingRad);
  }
  int rssi10 = constrain(WiFi.RSSI() * 10, -2000, 0);  // rover's own WiFi RSSI

  char payload[200];
  int len = snprintf(payload, sizeof(payload),
    "{\"id\":\"%s\",\"n\":\"%s\",\"t\":%lld,\"x\":%ld,\"y\":%ld,\"vx\":%ld,\"vy\":%ld,\"rssi\":%d}",
    RF_ID, nonce, tMs,
    lroundf(poseX * 10), lroundf(poseY * 10), lroundf(vx * 10), lroundf(vy * 10), rssi10);
  if (len > 0 && len < (int)sizeof(payload)) {
    sendSigned(laptop, PORT_REPLY, payload, len);
  }
}

void pollChallenges() {
  int size = udpIn.parsePacket();
  if (size <= 0) {
    return;
  }
  char buf[128];
  int n = udpIn.read(buf, sizeof(buf) - 1);
  if (n <= 0) {
    return;
  }
  buf[n] = 0;

  char nonce[33];
  long long tMs;
  if (parseChallenge(buf, nonce, tMs)) {
    laptopIP = udpIn.remoteIP();
    laptopKnown = true;
    answerChallenge(nonce, tMs, laptopIP);
  }
}

// ==================================================
// WIFI TELEMETRY (replaces the BLE status notify)
// ==================================================

void sendEncoderTelemetry() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  char payload[200];
  int len = snprintf(payload, sizeof(payload),
    "{\"id\":\"%s\",\"type\":\"tel\",\"seq\":%lu,\"cmd\":\"%c\",\"raw\":%d,\"rev_milli\":%ld,\"mag\":%d,"
    "\"x\":%ld,\"y\":%ld,\"h\":%ld,\"v\":%ld}",
    RF_ID, (unsigned long)telemetrySeq++, currentCommand,
    encoderDetected ? rawAngle : -1, lroundf(revolutions * 1000), magnetDetected ? 1 : 0,
    lroundf(poseX * 10), lroundf(poseY * 10), lroundf(headingRad * RAD_TO_DEG * 10), lroundf(speedCmS * 10));
  if (len <= 0 || len >= (int)sizeof(payload)) {
    return;
  }

  // to the laptop once it has challenged us; until then, broadcast on the hotspot subnet
  IPAddress target = laptopKnown ? laptopIP : IPAddress(255, 255, 255, 255);
  sendSigned(target, PORT_TELEMETRY, payload, len);
}

// ==================================================
// BLUETOOTH INITIALIZATION (control only)
// ==================================================

void initializeBluetooth() {
  BLEDevice::init("TURRETX-Rover");

  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);

  BLECharacteristic *commandCharacteristic =
      service->createCharacteristic(
        COMMAND_UUID,
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_WRITE_NR
      );

  commandCharacteristic->setCallbacks(new CommandCallbacks());

  // kept so existing controllers still find it; telemetry now goes over WiFi
  statusCharacteristic = service->createCharacteristic(
    STATUS_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_NOTIFY
  );

  statusCharacteristic->addDescriptor(new BLE2902());
  statusCharacteristic->setValue("READY - telemetry on WiFi");

  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);

  BLEDevice::startAdvertising();

  Serial.println("BLE initialized");
  Serial.println("Device name: TURRETX-Rover");
  Serial.println("Waiting for controller...");
}

// ==================================================
// WIFI INITIALIZATION
// ==================================================

void initializeWiFi() {
  WiFi.mode(WIFI_STA);
  // NOTE: do NOT call WiFi.setSleep(false): with BLE also running, the ESP32 requires WiFi
  // modem sleep and aborts otherwise. Replies get a few tens of ms of extra latency; fine.
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi: joining ");
  Serial.println(WIFI_SSID);
}

void serviceWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  if (!netStarted) {
    udpIn.begin(PORT_CHALLENGE);
    netStarted = true;
    Serial.print("WiFi connected, IP ");
    Serial.println(WiFi.localIP());
  }
  pollChallenges();
}

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("TURRETX ROVER: BLE CONTROL + WIFI DATA");
  Serial.println("================================");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(BOOT_BUTTON, INPUT_PULLUP);

  stopMotors();

  // MPU6050 bus: GPIO 17 and 18 (400 kHz for the 200 Hz gyro reads)
  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);
  Wire.setClock(400000);

  // AS5600 bus: GPIO 41 and 42
  EncoderWire.begin(ENC_SDA_PIN, ENC_SCL_PIN, 100000);

  initializeMPU6050();
  initializeAS5600();

  calibrateGyro();                              // rover must be still at the start pose
  updateEncoder();                              // first reading sets the encoder reference
  resetPose();

  initializeBluetooth();
  initializeWiFi();

  lastCommandAt = millis();
  lastMPUReadAt = millis();
  lastEncoderReadAt = millis();
  lastTelemetryAt = millis();
  lastGyroUs = micros();
  lastSpeedAt = millis();
}

// ==================================================
// MAIN LOOP
// ==================================================

void loop() {
  unsigned long now = millis();

  // Stop motors if browser commands stop arriving
  if (currentCommand != 'X' &&
      now - lastCommandAt > COMMAND_TIMEOUT_MS) {

    currentCommand = 'X';
    stopMotors();

    Serial.println("Command timeout - motors stopped");
  }

  updateHeading();

  if (now - lastMPUReadAt >= MPU_INTERVAL_MS) {
    lastMPUReadAt = now;
    readMPU6050();
  }

  if (now - lastEncoderReadAt >= ENCODER_INTERVAL_MS) {
    lastEncoderReadAt = now;
    updateEncoder();
  }

  if (now - lastTelemetryAt >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryAt = now;
    sendEncoderTelemetry();
  }

  if (digitalRead(BOOT_BUTTON) == LOW) {
    resetPose();
    delay(300);
  }

  serviceWiFi();

  delay(2);
}

import asyncio
from bleak import BleakScanner, BleakClient
import struct
import sys

uuid_gyro_service = '6fbe1da7-3002-44de-92c4-bb6e04fb0212'

def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    #output_numbers = list(data))
    #print(output_numbers)
    # print(', '.join('{:02x}'.format(x) for x in data))
    x = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[0:4])))[0], 2)
    y = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[4:8])))[0], 2)
    z = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[8:12])))[0], 2)
    print(str(x) + " \t" + str(y) + " \t" + str(z))

async def main():
    devices = await BleakScanner.discover()
    adress = ""
    found = False
    for d in devices:
        if d.name == "IMU1":
            adress = d.address
            found = True
        print(d)
    if found:
        async with BleakClient(adress) as client:
            print("Gyroscope found!")
            await client.start_notify(uuid_gyro_service, notification_handler)
            if(len(sys.argv) > 1 and sys.argv[1] == "endless"):
              while(1):
                  await asyncio.sleep(10.0)
            else:
                await asyncio.sleep(10.0)
            await client.stop_notify(uuid_gyro_service)
    else:
        print("Gyroscope not found!")

print("Searching for Gyroscope...")
asyncio.run(main())
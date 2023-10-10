import asyncio
from bleak import BleakScanner, BleakClient
import struct
import sys
import numpy

uuid_gyro_service = '6fbe1da7-3002-44de-92c4-bb6e04fb0212'

x_list = []
y_list = []
z_list = []
exciting_threshold = 30


def compute_class(std):
    return 2 if std < exciting_threshold else 0


def notification_handler(sender, data):
    global x_list, y_list, z_list
    """Simple notification handler which prints the data received."""
    #output_numbers = list(data))
    #print(output_numbers)
    # print(', '.join('{:02x}'.format(x) for x in data))
    x = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[0:4])))[0], 2)
    y = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[4:8])))[0], 2)
    z = round(struct.unpack('<f', bytes.fromhex(''.join('{:02x}'.format(x) for x in data[8:12])))[0], 2)
    print(str(x) + " \t" + str(y) + " \t" + str(z))
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)

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
            stdx = numpy.std(numpy.array(x_list))
            stdy = numpy.std(numpy.array(y_list))
            stdz = numpy.std(numpy.array(z_list))
            print("STD x-axis:", stdx, "STD y-axis:", stdy, "STD z_axis:", stdz)
            print("STD sum:", stdx+stdy+stdz, "Music class:", compute_class(stdx+stdy+stdz))
    else:
        print("Gyroscope not found!")

print("Searching for Gyroscope...")
asyncio.run(main())
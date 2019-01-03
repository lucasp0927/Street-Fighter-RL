s = manager:machine().screens[":screen"]
cpu = manager:machine().devices[":maincpu"]
mem = cpu.spaces["program"]
pause = false
-- ioports map
IN0 = manager:machine():ioport().ports[':IN0']
IN1 = manager:machine():ioport().ports[':IN1']
p1right = IN1.fields['P1 Right']
p1left = IN1.fields['P1 Left']
p1up = IN1.fields['P1 Up']
p1down = IN1.fields['P1 Down']
p1btn1 = IN1.fields['P1 Button 1']
p1btn2 = IN1.fields['P1 Button 2']
p1btn3 = IN0.fields['P1 Button 3']
p1btn4 = IN1.fields['P1 Button 4']
p1btn5 = IN1.fields['P1 Button 5']
p1btn6 = IN0.fields['P1 Button 6']
p2right = IN1.fields['P2 Right']
p2left = IN1.fields['P2 Left']
p2up = IN1.fields['P2 Up']
p2down = IN1.fields['P2 Down']
p2btn1 = IN1.fields['P2 Button 1']
p2btn2 = IN1.fields['P2 Button 2']
p2btn3 = IN0.fields['P2 Button 3']
p2btn4 = IN1.fields['P2 Button 4']
p2btn5 = IN1.fields['P2 Button 5']
p2btn6 = IN0.fields['P2 Button 6']
coin1 = IN0.fields['Coin 1']
coin2 = IN0.fields['Coin 2']
p1ctrl = {p1right,p1left,p1up,p1down,p1btn1,p1btn2,p1btn3,p1btn4,p1btn5,p1btn6}
p2ctrl = {p2right,p2left,p2up,p2down,p2btn1,p2btn2,p2btn3,p2btn4,p2btn5,p2btn6}


-- socket setting
socket = require("socket")
client = assert(socket.connect("192.168.0.105", 54321))
-- client = assert(socket.connect("*", 54321))
client:settimeout(0.016)
framecount = 0

-- HP values
-- HP = {0xFF86B6, 0xFF86E0} ryu ken
HP = {0xFF86B6, 0xFFC66A}

-- character frame buffers appear on top
top_frame = {0xFFE400,
             0xFFE440,
             0xFFE480,
             0xFFE4C0,
             0xFFE500,
             0xFFE540,
             0xFFE580,
             0xFFE6C0,
             0xFFE700,
             0xFFE740}

-- character frame buffers appear on bottom
bottom_frame = {0xFFEA00,
                0xFFEA40,
                0xFFEA80,
                0xFFEAc0,
                0xFFEB00,
                0xFFEB40,
                0xFFEB80,
                0xFFECC0,
                0xFFED00,
                0xFFED40}

-- Ryu's frame buffer values
ryu_types = {1,2,16,17,18,32,50,64,65,3,4,5,19,20,21,35,53,67,68,6,7,8,22,23,24,38,56,70,71,9,10,11,25,26,27,41,59,73,74,12,13,14,28,29,30,44,62,76,77,96,97,98,112,113,114,128,146,160,161,99,101,131,133,163,165,1,44,1544,1546,1576,1578,1608,1610,1540,1542,1572,1574,1604,1606,1548,1550,1580,1582,1612,1614,269,270,271,285,286,287,301,319,333,334,200,201,202,216,217,218,232,250,264,265,348,349,350,364,365,366,380,297,298,299,313,314,315,329,345,346,347,361,362,363,377,288,289,290,304,305,306,320,338,422,424,454,456,928,930,932,960,962,964,934,935,936,950,951,952,966,984,937,938,939,940,941,953,954,970,971,972,1355,1356,1357,1358,1359,1371,1372,1388,1389,1390,1292,1294,1324,1326,1288,1290,1320,1322,358,359,360,374,375,376,390,408,294,295,296,310,311,312,326,344,355,356,357,371,372,373,387,405,203,235,267,205,206,207,221,222,223,237,255,352,353,354,368,369,370,384,402,992,993,994,995,996,1008,1009,1025,1026,1027,1062,1064,1094,1096,1126,1128,844,846,876,878,908,910,840,842,872,874,904,906,836,838,868,870,900,902,742,744,746,774,776,778,806,832,834,864,866,896,898,419,420,421,435,436,437,451,469,1440,1442,1444,1472,1474,1476,1504,1536,1538,1568,1570,1600,1602,1446,1448,1478,1480,1510,1512,748,750,780,782,812,814,1188,1190,1220,1222,1252,1254,1192,1194,1224,1226,1256,1258,1066,1068,1070,1098,1100,1102,1130,488,8140,8142,8172,8174,997,998,999,1000,1001,1013,1014,1030,1031,1032,640,642,672,674,704,706,648,650,680,682,712,714,644,646,676,678,708,710,554,556,558,586,588,590,618,1418,1420,1422,1450,1452,1454,1482,394,396,398,426,428,430,458,416,417,418,432,433,434,448,466,1002,1003,1004,1005,1006,1018,1019,1035,1036,1037,192,194,224,226,256,258,196,198,228,230,260,262,1056,1058,1088,1090,1120,1122,1344,1346,1376,1378,1408,1410,1152,1154,1184,1186,1156,1158,1160,1162,491,492,493,494,495,507,508,524,525,526,291,292,293,307,308,309,323,341,544,546,576,578,608,610,548,550,552,580,582,584,612,652,653,654,668,669,670,684,702,716,717,736,738,740,768,770,772,800,1284,1286,1316,1318,1348,1350,1164,1166,1196,1198,1228,1230,1260,1216,1218,1248,1250,1280,1282,1312}

-- frame buffer for "Hadouken"
projectile_frame = {0xFFE100}

a = 16;
offset_x = -50;
offset_y = 0;

function toBits(num)
    -- returns a table of bits, least significant first.
    local t={} -- will contain the bits
    while num>=0 do
        rest=math.fmod(num,2)
        t[#t+1]=1-math.floor(rest)
        num=(num-rest)/2
        if num == 0 then break end
    end
    return table.concat(t)
end

function has_value (tab, val)
    for index, value in ipairs (tab) do
        if value == val then
            return true
        end
    end
    return false
end

function top_is_ryu()
    if has_value(ryu_types,mem:read_i16(top_frame[1])) then
        return true
    end
    return false
end

-- visualization of frame buffer block
function draw_framebox(addr, color)
   local x = mem:read_i16(addr+6)
   local y = mem:read_i16(addr+4)
   local stype = mem:read_i16(addr)
   local dir = mem:read_i8(addr+2)
   local b = 0
   if dir == 0 or dir == 1 then
      b = a/2;
   elseif dir == 4 or dir ==5 then
      b = a;
   end
   s:draw_box(x+offset_x-b, y+offset_y-b, x+offset_x+b, y+offset_y+b, 0, color); -- (x0, y0, x1, y1, fill-color, line-color)
   s:draw_text(x+offset_x, y+offset_y, tostring(stype).." "..tostring(dir)); -- (x0, y0, msg)
end

function heal_hp()
    mem:write_u16(0xFF86B6, 0x040B)
    mem:write_u16(0xFFC66A, 0x040B)
end
-- message to send
function msg()
    local msg = {}
    local reward = (mem:read_i8(0xFF86B7)+1/4*(mem:read_i8(0xFF86B6))) - (mem:read_i8(0xFFC66B)+1/4*(mem:read_i8(0xFFC66A)))
    heal_hp()
    table.insert(msg, reward)
    if top_is_ryu() then
        -- for _, v in pairs(HP) do
        --     table.insert(msg, mem:read_i8(v+1)+1/4*(mem:read_i8(v)))
        -- end
        for _, v in pairs(top_frame) do
            table.insert(msg, mem:read_i16(v+6))
            table.insert(msg, mem:read_i16(v+4))
            table.insert(msg, mem:read_i16(v))
        end
        for _, v in pairs(bottom_frame) do
            table.insert(msg, mem:read_i16(v+6))
            table.insert(msg, mem:read_i16(v+4))
            table.insert(msg, mem:read_i16(v))
        end
    else
        -- for _, v in pairs(HP) do
        --     table.insert(msg, mem:read_i8(v+1)+1/4*(mem:read_i8(v)))
        -- end
        for _, v in pairs(bottom_frame) do
            table.insert(msg, mem:read_i16(v+6))
            table.insert(msg, mem:read_i16(v+4))
            table.insert(msg, mem:read_i16(v))
        end
        for _, v in pairs(top_frame) do
            table.insert(msg, mem:read_i16(v+6))
            table.insert(msg, mem:read_i16(v+4))
            table.insert(msg, mem:read_i16(v))
        end
    end
    table.insert(msg, mem:read_i16(projectile_frame[1]+6))
    table.insert(msg, mem:read_i16(projectile_frame[1]+4))
    table.insert(msg, mem:read_i16(projectile_frame[1]))
    return table.concat(msg, " ")
end

-- ioport trigger
function input_dir(dir)
    if dir == 1 then --up
        p1up:set_value(1)
    elseif dir == 2 then --up right
        p1up:set_value(1)
        p1right:set_value(1)
    elseif dir == 3 then --right
        p1right:set_value(1)
    elseif dir == 4 then --down right
        p1down:set_value(1)
        p1right:set_value(1)
    elseif dir == 5 then --down
        p1down:set_value(1)
    elseif dir == 6 then --down left
        p1down:set_value(1)
        p1left:set_value(1)
    elseif dir == 7 then --left
        p1left:set_value(1)
    elseif dir == 8 then --up left
        p1up:set_value(1)
        p1left:set_value(1)
    end
end
function input_attack(attack)
    if attack == 1 then
        p1btn1:set_value(1)
    elseif attack == 2 then
        p1btn2:set_value(1)
    elseif attack == 3 then
        p1btn3:set_value(1)
    elseif attack == 4 then
        p1btn4:set_value(1)
    elseif attack == 5 then
        p1btn5:set_value(1)
    elseif attack == 6 then
        p1btn6:set_value(1)
    end
end
function clear_input()
    for k,v in pairs(p1ctrl) do
        v:set_value(0)
    end
end
function IOinput(msg)
    --print(msg)
    clear_input()
    local dir = tonumber(string.sub(msg,1,1))
    --print("dir "..tostring(dir))
    input_dir(dir)
    local attack = tonumber(string.sub(msg,2,2))
    --print("attack "..tostring(attack))
    input_attack(attack)
    --for k,v in pairs(p1ctrl) do
    --    v:set_value(tonumber(string.sub(msg,k,k)))
    --end
end

-- order of io buttons our's -> lua's
--p1ctrl = {p1right,p1left,p1up,p1down,p1btn1,p1btn2,p1btn3,p1btn4,p1btn5,p1btn6}
--p2ctrl = {p2right,p2left,p2up,p2down,p2btn1,p2btn2,p2btn3,p2btn4,p2btn5,p2btn6}
io1_orders = {1,2,4,3,5,6,26,7,8,19}
io2_orders = {9,10,12,11,13,14,27,15,16,25}

function IOoutput()
      local out1 = {}
      local msg = ""
--      out2 = ""
      local intbl = toBits(manager:machine():ioport().ports[':IN1']:read()) .. toBits(manager:machine():ioport().ports[':IN0']:read())
      for k,v in pairs(io1_orders) do
          out1[k] = tonumber(string.sub(intbl,v,v))
      end
      if out1[3] == 1 and out1[1]==0 and out1[2]==0 then--up
          msg = "1"
      elseif out1[3]==1 and out1[1]==1 then --up right
          msg = "2"
      elseif out1[1]==1 and out1[3]==0 and out1[4]==0 then --right
          msg = "3"
      elseif out1[1]==1 and out1[4]==1 then --down right
          msg = "4"
      elseif out1[4]==1 and out1[1]==0 and out1[2]==0 then --down
          msg = "5"
      elseif out1[4]==1 and out1[2]==1 then --down left
          msg = "6"
      elseif out1[2]==1 and out1[3]==0 and out1[4]==0 then --left
          msg = "7"
      elseif out1[3]==1 and out1[2]==1 then -- left up
          msg = "8"
      elseif out1[1]==0 and out1[2]==0 and out1[3]==0 and out1[4]==0 then
          msg = "0"
      else
          print("wrong direction combination.")
      end
      if out1[5]==1 then
          msg = msg.." ".."1"
      elseif out1[6]==1 then
          msg = msg.." ".."2"
      elseif out1[7]==1 then
          msg = msg.." ".."3"
      elseif out1[8]==1 then
          msg = msg.." ".."4"
      elseif out1[9]==1 then
          msg = msg.." ".."5"
      elseif out1[10]==1 then
          msg = msg.." ".."6"
      elseif out1[5]==0 and out1[6]==0 and out1[7]==0 and out1[8]==0 and out1[9]==0 and out1[10]==0 then
          msg = msg.." ".."0"
      end
    --   for k,v in pairs(io2_orders) do
    --       out2 = out2 .. string.sub(intbl,v,v)
    --   end
      return msg
    --   print(out2)

end

-- frame hook
function draw_hud()
   framecount = framecount + 1
   if framecount == 2 then framecount = 0 end
   --client:send(tonumber(framecount).."\n")
   mem:write_u32(0xFF8E16, 0xFFFFFFFF); --time
   --mem:write_u16(0xFF86B6, 0x000B); --HP
   --mem:write_u16(0xFF86E0, 0x000B); --HP
   local p1hp = mem:read_i8(0xFF86B7)+1/4*(mem:read_i8(0xFF86B6))
   -- local p2hp = mem:read_u16(0xFF86E0)
   local p2hp = mem:read_i8(0xFFC66B)+1/4*(mem:read_i8(0xFFC66A))
   s:draw_text(50, 10 , tostring(p1hp));
   s:draw_text(250, 10, tostring(p2hp));
   --local line, err = client:receive()
   --print(line)
   if top_is_ryu() then
      for index, value in pairs(top_frame) do draw_framebox(value, 0xffffffff) end
      for index, value in pairs(bottom_frame) do draw_framebox(value, 0xffff00ff) end
   else
      for index, value in pairs(top_frame) do draw_framebox(value, 0xffff00ff) end
      for index, value in pairs(bottom_frame) do draw_framebox(value, 0xffffffff) end
   end
   for index, value in pairs(projectile_frame) do
       draw_framebox(value, 0xffffff00)
   end
   if framecount == 0 and mem:read_i16(0xFFE406) ~= 0 then
       if pause == false then
           client:send(msg() .." " .. IOoutput())
       else
           heal_hp()
       end
       local line, err = client:receive()
       if line == "pause" then
           print("pause game")
           IOinput("00")
           pause = true
       elseif line == "unpause" then
           print("resume game")
           pause = false
       elseif line ~= nil then
           IOinput(line)
       end
   --IOoutput()
   end


   -- heal both characters to full HP

end


emu.sethook(draw_hud, "frame")

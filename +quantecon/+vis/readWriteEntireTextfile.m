function fstrm = readWriteEntireTextfile(fname, fstrm)
%READWRITEENTIRETEXTFILE Read or write a whole text file to/from memory
%
%   Ported to quantecon.vis.

modes = {'rt', 'wt'};
writing = nargin > 1;
fh = fopen(char(fname), modes{1 + writing});
if fh == -1
    error('Unable to open file %s.', char(fname));
end
try
    if writing
        fwrite(fh, fstrm, 'char*1');
    else
        fstrm = fread(fh, '*char')';
    end
catch ex
    fclose(fh);
    rethrow(ex);
end
fclose(fh);
end
